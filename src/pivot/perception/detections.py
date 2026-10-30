"""Per-frame instance detections and their grouping into tracks.

Both perception backends, the Meta Model API and a self-hosted SAM 3.1,
produce these detections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pivot.geometry.types import BoolArray
from pivot.perception.tracks import Track


@dataclass(frozen=True)
class Detection:
    """One object instance in one frame.

    Attributes:
        object_id: Identifier, stable across the frames of one response.
        frame: Zero-based frame index (0 for images).
        bounds: Half-open box ``(left, top, right, bottom)`` in source pixels.
        mask: Box-local mask of shape ``(bottom - top, right - left)``.
        confidence: Detection confidence in [0, 1], if requested.
    """

    object_id: int
    frame: int
    bounds: tuple[int, int, int, int]
    mask: BoolArray
    confidence: float | None = None

    def full_mask(self, width: int, height: int) -> BoolArray:
        """Pastes the box-local mask into a full-frame mask.

        Args:
            width: Frame width, in pixels.
            height: Frame height, in pixels.

        Returns:
            A (height, width) mask.
        """
        left, top, right, bottom = self.bounds
        full = np.zeros((height, width), dtype=bool)
        # Clip in case the box touches the frame border.
        rows, cols = min(bottom, height) - top, min(right, width) - left
        full[top:top + rows, left:left + cols] = self.mask[:rows, :cols]
        return full


def tracks_from_detections(detections: list[Detection], label: str, width: int, height: int,
                           frame_map: dict[int, int] | None = None,
                           first_id: int = 0) -> list[Track]:
    """Groups detections into one track per SAM object id.

    Args:
        detections: Detections of one response.
        label: The noun phrase the response was prompted with.
        width: Frame width, in pixels.
        height: Frame height, in pixels.
        frame_map: Optional mapping from video frame index to reconstruction
            frame index; frames absent from the map are dropped.
        first_id: Track ids are assigned consecutively from this value.

    Returns:
        Tracks sorted by SAM object id.
    """
    by_object: dict[int, list[Detection]] = {}
    for detection in detections:
        by_object.setdefault(detection.object_id, []).append(detection)
    tracks = []
    for offset, object_id in enumerate(sorted(by_object)):
        track = Track(first_id + offset, label)
        scores = []
        for detection in by_object[object_id]:
            frame = detection.frame if frame_map is None else frame_map.get(detection.frame)
            if frame is None:
                continue
            track.masks[frame] = detection.full_mask(width, height)
            if detection.confidence is not None:
                scores.append(detection.confidence)
        track.score = float(np.mean(scores)) if scores else 1.0
        tracks.append(track)
    return tracks
