"""Joining tracks across a cut of the video by following pixels with CoTracker3.

When a video is tracked in pieces (see :mod:`pivot.perception.chunks`), an
object hidden on the frames two pieces share gets a new id after the cut.
Meta's CoTracker3 follows points through frames: points sampled in the
object's last mask before the cut are followed to the first frame of a
candidate track after it; the share of them that lands in its mask says
whether it is the same object. CoTracker3 needs ``torch`` (a GPU helps; a
few short clips also run on the CPU) and downloads its weights from
``facebookresearch/co-tracker`` on first use.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pivot.data.video import read_frames
from pivot.geometry.types import FloatArray
from pivot.perception.chunks import Linker
from pivot.perception.detections import Detection

# Follows points through frames: (T, H, W, 3) RGB frames and (N, 2) pixel
# points of the first frame to their (T, N, 2) positions and (T, N) visibility.
PointTracker = Callable[[npt.NDArray[np.uint8], FloatArray],
                        tuple[FloatArray, npt.NDArray[np.bool_]]]


def mask_points(detection: Detection, count: int) -> FloatArray:
    """Samples pixels of a detection's mask, spread over it.

    Args:
        detection: The detection.
        count: Most points.

    Returns:
        (N, 2) pixel coordinates ``(x, y)`` in the frame, N <= count.
    """
    rows, cols = np.nonzero(detection.mask)
    if len(rows) == 0:
        return np.zeros((0, 2))
    picked = np.linspace(0, len(rows) - 1, min(count, len(rows))).round().astype(int)
    left, top = detection.bounds[0], detection.bounds[1]
    return np.column_stack([cols[picked] + left + 0.5, rows[picked] + top + 0.5]).astype(float)


def inside(detection: Detection, points: FloatArray) -> npt.NDArray[np.bool_]:
    """Tests which pixel points fall in a detection's mask.

    Args:
        detection: The detection.
        points: (N, 2) pixel coordinates ``(x, y)``.

    Returns:
        (N,) True for points in the mask.
    """
    left, top, right, bottom = detection.bounds
    cols, rows = np.floor(points[:, 0]).astype(int), np.floor(points[:, 1]).astype(int)
    result = (cols >= left) & (cols < right) & (rows >= top) & (rows < bottom)
    result[result] = detection.mask[rows[result] - top, cols[result] - left]
    return np.asarray(result, dtype=bool)


def point_linker(video: Path, size: tuple[int, int], tracker: PointTracker,
                 count: int = 48, read: Callable[..., npt.NDArray[np.uint8]] = read_frames
                 ) -> Linker:
    """Builds a linker that follows the pixels of an object across frames.

    Args:
        video: The whole video the detections' frames refer to.
        size: Its frame size ``(width, height)``.
        tracker: Follows points through frames (CoTracker3, see :func:`load_cotracker3`).
        count: Points sampled in the object's last mask.
        read: Decodes frames (see :func:`~pivot.data.video.read_frames`); for tests.

    Returns:
        A linker: the share of the followed points, still visible at the
        other track's first frame, that land in its mask.
    """
    def link(last: Detection, first: Detection) -> float:
        """Scores the continuation of one track by another.

        Args:
            last: Last detection of the earlier track.
            first: First detection of the later track.

        Returns:
            The share of followed points landing in the later mask.
        """
        points = mask_points(last, count)
        if len(points) == 0:
            return 0.0
        frames = read(video, last.frame, first.frame, *size)
        tracks, visible = tracker(frames, points)
        seen = visible[-1]
        if not seen.any():
            return 0.0
        return float(inside(first, tracks[-1][seen]).mean())
    return link


def load_cotracker3(device: str | None = None) -> PointTracker | None:
    """Loads Meta's CoTracker3 (offline model), if ``torch`` is installed.

    Args:
        device: Torch device; the GPU if there is one, else the CPU.

    Returns:
        The point tracker, or None without ``torch`` or the weights.
    """
    try:
        # Optional: the API pipeline runs without torch, joining tracks by mask overlap only.
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError:
        return None
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model: Any = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
    except Exception:  # pylint: disable=broad-exception-caught
        # No network or no weights: the tracks are joined by mask overlap only.
        return None
    model = model.to(device).eval()

    def track(frames: npt.NDArray[np.uint8], points: FloatArray
              ) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
        """Follows points with CoTracker3.

        Args:
            frames: (T, H, W, 3) RGB frames.
            points: (N, 2) pixel points of the first frame.

        Returns:
            Their (T, N, 2) positions and (T, N) visibility.
        """
        clip = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)[None]
        queries = torch.from_numpy(np.column_stack([np.zeros(len(points)), points]))[None]
        with torch.no_grad():
            tracks, visible = model(clip.float().to(device), queries=queries.float().to(device))
        return (tracks[0].cpu().numpy().astype(np.float64),
                visible[0].cpu().numpy().astype(bool))
    return track
