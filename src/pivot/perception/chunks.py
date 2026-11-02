"""Tracking a long video in parallel pieces, and joining the pieces' tracks.

SAM 3.1 runs through a video frame after frame: one request on 2,700 frames
takes about fifteen minutes, whatever the parallelism across kinds of
objects. Cut into pieces that overlap by a few frames, the video is tracked
piece by piece at the same time, and the tracks are joined where the pieces
overlap: on the shared frames, the same object gets nearly the same mask
from both pieces, so tracks are matched by mask overlap (IoU). An object no
shared frame shows (hidden right at the cut) can be joined by following its
pixels across the cut with a point tracker (CoTracker3); tracks left apart
are still merged later in 3D when they never meet in one frame.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Callable

import numpy as np

from pivot.perception.detections import Detection

# Each video request is billed for at least this many frames.
MIN_BILLED_FRAMES = 50
# Shared frames of two consecutive pieces.
OVERLAP = 8
# Mean mask overlap of two tracks on their shared frames for them to be one object.
MIN_IOU = 0.5

# Says how likely a track ending with the first detection is to continue as
# the track starting with the second one, in [0, 1].
Linker = Callable[[Detection, Detection], float]


def plan_chunks(frames: list[int], length: int, overlap: int = OVERLAP,
                min_frames: int = MIN_BILLED_FRAMES) -> list[list[int]]:
    """Cuts a list of frames into overlapping pieces.

    Args:
        frames: Frame indices of the video to track, in order.
        length: Most frames of one piece.
        overlap: Frames shared by two consecutive pieces.
        min_frames: Fewest frames of a piece (shorter ones are billed as this
            many): a short tail joins the piece before.

    Returns:
        The pieces, as lists of frame indices; one piece if the video is short.

    Raises:
        ValueError: If a piece would be no longer than the overlap.
    """
    if length <= overlap:
        raise ValueError("A piece must be longer than the overlap.")
    chunks: list[list[int]] = []
    start = 0
    while True:
        end = min(start + length, len(frames))
        if len(frames) - end < min_frames:
            end = len(frames)
        chunks.append(frames[start:end])
        if end == len(frames):
            return chunks
        start = end - overlap


def mask_iou(a: Detection, b: Detection) -> float:
    """Measures the overlap of two detections' masks, in the same frame.

    Args:
        a: A detection.
        b: Another one.

    Returns:
        Intersection over union, in [0, 1].
    """
    left, top = min(a.bounds[0], b.bounds[0]), min(a.bounds[1], b.bounds[1])
    right, bottom = max(a.bounds[2], b.bounds[2]), max(a.bounds[3], b.bounds[3])
    canvases = []
    for d in (a, b):
        canvas = np.zeros((bottom - top, right - left), dtype=bool)
        rows, cols = d.mask.shape
        canvas[d.bounds[1] - top:d.bounds[1] - top + rows,
               d.bounds[0] - left:d.bounds[0] - left + cols] = d.mask
        canvases.append(canvas)
    union = int(np.logical_or(*canvases).sum())
    return float(np.logical_and(*canvases).sum()) / union if union else 0.0


def _by_object(detections: list[Detection]) -> dict[int, dict[int, Detection]]:
    """Groups detections by object, then by frame.

    Args:
        detections: Detections.

    Returns:
        ``{object_id: {frame: detection}}``.
    """
    grouped: dict[int, dict[int, Detection]] = defaultdict(dict)
    for d in detections:
        grouped[d.object_id][d.frame] = d
    return grouped


def _match(before: dict[int, dict[int, Detection]], after: dict[int, dict[int, Detection]],
           shared: set[int], min_iou: float) -> dict[int, int]:
    """Matches the tracks of two pieces by their mean mask overlap on shared frames.

    Args:
        before: Tracks of the earlier piece, by global id then frame.
        after: Tracks of the later piece, by local id then frame.
        shared: Frames both pieces cover.
        min_iou: Smallest mean overlap of a match.

    Returns:
        ``{local id: global id}`` of the matched tracks, one to one.
    """
    scores = []
    for global_id, old in before.items():
        for local_id, new in after.items():
            frames = [f for f in shared if f in old or f in new]
            if not frames or not any(f in old and f in new for f in frames):
                continue
            iou = sum(mask_iou(old[f], new[f]) if f in old and f in new else 0.0
                      for f in frames) / len(frames)
            if iou >= min_iou:
                scores.append((iou, local_id, global_id))
    matched: dict[int, int] = {}
    for _, local_id, global_id in sorted(scores, reverse=True):
        if local_id not in matched and global_id not in matched.values():
            matched[local_id] = global_id
    return matched


def _link_across(before: dict[int, dict[int, Detection]], after: dict[int, dict[int, Detection]],
                 matched: dict[int, int], linker: Linker, max_gap: int) -> None:
    """Joins tracks that no shared frame shows, with a point tracker.

    Args:
        before: Tracks of the earlier pieces, by global id then frame.
        after: Tracks of the later piece, by local id then frame.
        matched: ``{local id: global id}``, completed in place.
        linker: Scores the continuation of one track by another.
        max_gap: Most frames between the end of one and the start of the other.
    """
    scores = []
    for local_id, new in after.items():
        if local_id in matched:
            continue
        first = new[min(new)]
        for global_id, old in before.items():
            if global_id in matched.values():
                continue
            last = old[max(old)]
            if 0 < first.frame - last.frame <= max_gap:
                scores.append((linker(last, first), local_id, global_id))
    for score, local_id, global_id in sorted(scores, reverse=True):
        if score >= 0.5 and local_id not in matched and global_id not in matched.values():
            matched[local_id] = global_id


# pylint: disable-next=too-many-locals
def join_chunks(pieces: list[list[Detection]], chunks: list[list[int]],
                min_iou: float = MIN_IOU, linker: Linker | None = None,
                max_gap: int = 30) -> list[Detection]:
    """Joins the detections of the pieces of a video into tracks of the whole video.

    Args:
        pieces: Detections of each piece, frames counted within the piece.
        chunks: Frame indices of each piece in the whole video (see :func:`plan_chunks`).
        min_iou: Smallest mean mask overlap on shared frames of one object.
        linker: Joins tracks no shared frame shows; None leaves them apart.
        max_gap: Most frames a linked track may skip.

    Returns:
        The detections, frames counted in the whole video, ids numbered from 1;
        on shared frames, a joined track keeps the earlier piece's detections.
    """
    tracks: dict[int, dict[int, Detection]] = {}
    covered: set[int] = set()
    for piece, frames in zip(pieces, chunks):
        after = _by_object([dataclasses.replace(d, frame=frames[d.frame]) for d in piece])
        matched = _match(tracks, after, covered & set(frames), min_iou)
        if linker is not None:
            _link_across(tracks, after, matched, linker, max_gap)
        for local_id in sorted(after):
            if local_id not in matched:
                matched[local_id] = max([*tracks, *matched.values()], default=0) + 1
        for local_id, detections in after.items():
            track = tracks.setdefault(matched[local_id], {})
            for frame, d in detections.items():
                track.setdefault(frame, dataclasses.replace(d, object_id=matched[local_id]))
        covered |= set(frames)
    return sorted((d for track in tracks.values() for d in track.values()),
                  key=lambda d: (d.frame, d.object_id))
