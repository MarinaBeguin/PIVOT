"""Object tracks: per-frame instance masks sharing one identity."""

from dataclasses import dataclass, field

from pivot.geometry.types import BoolArray


@dataclass
class Track:
    """One object followed across the frames of a video.

    Attributes:
        track_id: Identifier, unique within a set of tracks.
        label: Noun phrase the object was segmented with, e.g. ``"chair"``.
        masks: Instance masks keyed by frame index. Masks may have any
            resolution; consumers resize them as needed.
        score: Mean detection confidence in [0, 1].
    """

    track_id: int
    label: str
    masks: dict[int, BoolArray] = field(default_factory=dict)
    score: float = 1.0
