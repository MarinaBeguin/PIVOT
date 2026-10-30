"""Tests for the Meta Model API layer, without network access."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from openai import NotFoundError

from pivot.meta.client import MetaConfig, call_with_retry, is_spurious_not_found
from pivot.meta.sam import SamClient, parse_sam_output
from pivot.perception.detections import Detection, tracks_from_detections

DATA = Path(__file__).parent / "data"


def not_found(code: str = "model_not_found") -> NotFoundError:
    request = httpx.Request("POST", "https://api.meta.ai/v1/responses")
    body = {"code": code, "message": "The requested model was not found."}
    return NotFoundError("Error code: 404", response=httpx.Response(404, request=request),
                         body=body)


def test_retry_recovers_from_spurious_not_found() -> None:
    outcomes: list[Any] = [not_found(), not_found(), "ok"]
    delays: list[float] = []

    def call() -> str:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return str(outcome)

    assert call_with_retry(call, sleep=delays.append) == "ok"
    assert delays == [0.5, 1.0]


def test_retry_gives_up_after_all_attempts() -> None:
    def call() -> str:
        raise not_found()

    with pytest.raises(NotFoundError):
        call_with_retry(call, attempts=3, sleep=lambda _: None)


def test_other_not_found_errors_are_not_retried() -> None:
    assert is_spurious_not_found(not_found())
    assert not is_spurious_not_found(not_found(code="file_not_found"))
    assert not is_spurious_not_found(ValueError("model_not_found"))


def test_config_requires_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("META_API_KEY", raising=False)
    monkeypatch.delenv("PIVOT_BACKEND", raising=False)
    with pytest.raises(RuntimeError):
        MetaConfig.from_env(tmp_path / "missing.env")


def test_open_version_needs_no_key_and_targets_local_llama(monkeypatch: pytest.MonkeyPatch,
                                                           tmp_path: Path) -> None:
    for name in ("META_API_KEY", "PIVOT_AGENT_MODEL", "PIVOT_REASONING_EFFORT", "PIVOT_OPEN_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PIVOT_BACKEND", "open")
    config = MetaConfig.from_env(tmp_path / "missing.env")
    assert config.base_url == "http://localhost:8000/v1"
    assert config.agent_model.startswith("meta-llama/")
    assert config.reasoning_effort is None
    monkeypatch.setenv("PIVOT_OPEN_URL", "http://octopus04:8000/v1")
    assert MetaConfig.from_env(tmp_path / "missing.env").base_url == "http://octopus04:8000/v1"


def test_config_reads_model_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("META_API_KEY", "test")
    monkeypatch.setenv("PIVOT_AGENT_MODEL", "muse-wonder-1.0")
    config = MetaConfig.from_env(tmp_path / "missing.env")
    assert config.agent_model == "muse-wonder-1.0"
    assert config.sam_model == "sam-3.1"


def test_parse_real_image_response() -> None:
    # Recorded SAM 3.1 answer for "chair" on a 640 x 480 ScanNet++ frame.
    detections = parse_sam_output((DATA / "sam_image_chairs.txt").read_text(), "image")
    assert [d.object_id for d in detections] == [0, 1]
    first = detections[0]
    assert first.bounds == (525, 127, 640, 326)
    assert first.mask.shape == (199, 115)
    assert 0.2 < first.mask.mean() < 0.6


def test_full_mask_pastes_box_local_mask() -> None:
    mask = np.ones((2, 3), dtype=bool)
    full = Detection(7, 0, (4, 1, 7, 3), mask).full_mask(width=6, height=5)
    assert full.sum() == 4  # Two columns fall outside the 6-pixel-wide frame.
    assert full[1:3, 4:6].all()


def test_tracks_group_by_object_and_map_frames() -> None:
    mask = np.ones((1, 1), dtype=bool)
    detections = [Detection(3, 0, (0, 0, 1, 1), mask, 0.8),
                  Detection(5, 0, (1, 1, 2, 2), mask, 0.6),
                  Detection(3, 10, (0, 0, 1, 1), mask, 0.6)]
    tracks = tracks_from_detections(detections, "chair", 4, 4, frame_map={0: 0, 10: 1},
                                    first_id=100)
    assert [t.track_id for t in tracks] == [100, 101]
    assert sorted(tracks[0].masks) == [0, 1]
    assert tracks[0].score == pytest.approx(0.7)
    assert tracks[0].masks[1].shape == (4, 4)


class FakeStream:
    """Context manager mimicking ``client.responses.stream``."""

    def __init__(self, text: str):
        self._events = [SimpleNamespace(type="response.created"),
                        SimpleNamespace(type="response.output_text.delta", delta=text)]

    def __enter__(self) -> list[SimpleNamespace]:
        return self._events

    def __exit__(self, *args: object) -> None:
        return None


def test_sam_client_caches_responses(tmp_path: Path) -> None:
    text = (DATA / "sam_image_chairs.txt").read_text()
    calls: list[dict[str, Any]] = []

    def stream(**kwargs: Any) -> FakeStream:
        calls.append(kwargs)
        return FakeStream(text)

    fake = SimpleNamespace(responses=SimpleNamespace(stream=stream))
    image = tmp_path / "frame.jpg"
    image.write_bytes(b"\xff\xd8 not really a jpeg")
    config = MetaConfig(api_key="test")
    sam = SamClient(config, tmp_path / "cache", client=fake)  # type: ignore[arg-type]
    first = sam.segment_image(image, "chair")
    second = sam.segment_image(image, "Chair ")
    assert len(calls) == 1
    assert calls[0]["model"] == "sam-3.1"
    assert [d.bounds for d in first] == [d.bounds for d in second]
