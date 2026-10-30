"""Tests for loading detection files into tracks before building a scene."""

from pathlib import Path

import numpy as np

from pivot.build import load_tracks
from pivot.perception.detections import Detection
from pivot.perception.sam3_local import save_detections


def test_load_tracks_keeps_the_requested_prompts_only(tmp_path: Path) -> None:
    for name in ("chair", "trash_can", "desk"):
        save_detections([Detection(1, 0, (0, 0, 2, 2), np.ones((2, 2), dtype=bool), 0.9)],
                        tmp_path / f"{name}.npz")
    every = load_tracks(tmp_path, 8, (8, 6))
    assert sorted(t.label for t in every) == ["chair", "desk", "trash can"]
    chosen = load_tracks(tmp_path, 8, (8, 6), ["trash can", "chair"])
    assert [t.label for t in chosen] == ["trash can", "chair"]
    assert [t.track_id for t in chosen] == [1, 2]
