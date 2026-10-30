"""Tests for the self-hosted SAM 3.1 backend's storage and output handling."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from pivot.perception.detections import Detection
from pivot.perception.sam3_local import (
    detection_from_mask,
    drop_unsupported_kwargs,
    load_detections,
    progress_path,
    prompt_filename,
    save_detections,
    track_prompt,
    track_prompts,
    write_progress,
)


def test_round_trip_preserves_masks(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    detections = [Detection(3, 7, (10, 20, 15, 23), rng.random((3, 5)) > 0.5, 0.9),
                  Detection(4, 8, (0, 0, 2, 1), np.array([[True, False]]), None)]
    path = tmp_path / "chair.npz"
    save_detections(detections, path)
    restored = load_detections(path)
    assert [(d.object_id, d.frame, d.bounds) for d in restored] == \
        [(d.object_id, d.frame, d.bounds) for d in detections]
    for original, loaded in zip(detections, restored):
        np.testing.assert_array_equal(original.mask, loaded.mask)
    assert restored[0].confidence == 0.9 and restored[1].confidence is None


def test_round_trip_of_no_detections(tmp_path: Path) -> None:
    save_detections([], tmp_path / "none.npz")
    assert not load_detections(tmp_path / "none.npz")


def test_detection_from_mask_crops_to_box() -> None:
    mask = np.zeros((6, 8), dtype=bool)
    mask[2:4, 3:7] = True
    detection = detection_from_mask(1, 0, mask, 0.5)
    assert detection is not None
    assert detection.bounds == (3, 2, 7, 4)
    np.testing.assert_array_equal(detection.full_mask(8, 6), mask)
    assert detection_from_mask(1, 0, np.zeros((2, 2), dtype=bool), None) is None


def test_prompt_filename_is_safe() -> None:
    assert prompt_filename(" Trash can/bin ") == "trash_can_bin.npz"


class FakePredictor:
    """Mimics the request API of SAM 3.1's multiplex video predictor."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(request)
        return {}

    def handle_stream_request(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self.requests.append(request)
        for frame in range(2):
            masks = np.zeros((2, 4, 4), dtype=bool)
            masks[0, frame:frame + 2, :2] = True  # Object 5 is visible, object 9 is not.
            yield {"frame_index": frame, "outputs": {
                "out_obj_ids": np.array([5, 9]), "out_probs": np.array([0.8, 0.1]),
                "out_binary_masks": masks}}


def test_track_prompt_resets_session_and_collects_frames() -> None:
    predictor = FakePredictor()
    frames: list[int] = []
    detections = track_prompt(predictor, "s1", "chair", frames.append)
    assert frames == [0, 1]
    assert [r["type"] for r in predictor.requests] == \
        ["reset_session", "add_prompt", "propagate_in_video"]
    assert predictor.requests[-1]["evict_cached_frame_outputs"]
    assert [(d.object_id, d.frame) for d in detections] == [(5, 0), (5, 1)]
    assert detections[1].bounds == (0, 1, 2, 3)


def test_drop_unsupported_kwargs_filters_by_signature() -> None:
    def init_state(resource_path: str, offload_video_to_cpu: bool = False) -> tuple[str, bool]:
        return resource_path, offload_video_to_cpu

    wrapped = drop_unsupported_kwargs(init_state)
    assert wrapped(resource_path="a.mp4", offload_video_to_cpu=True,
                   offload_state_to_cpu=False) == ("a.mp4", True)


def test_progress_is_a_permille_removed_once_done(tmp_path: Path) -> None:
    path = progress_path(tmp_path, "trash can")
    assert path.name == "trash_can.progress"
    write_progress(path, 0.4567)
    assert path.read_text(encoding="utf-8") == "457"

    def track(prompt: str) -> list[Detection]:
        write_progress(progress_path(tmp_path, prompt), 0.5)
        return []

    track_prompts(["trash can"], tmp_path, track)
    assert (tmp_path / "trash_can.npz").exists() and not path.exists()
