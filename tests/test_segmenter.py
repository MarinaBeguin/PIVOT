"""Tests for segmentation backend selection."""

from pathlib import Path

import httpx
import numpy as np
import pytest
from openai import NotFoundError

from pivot.perception.detections import Detection
from pivot.perception.sam3_local import save_detections
from pivot.perception.segmenter import Backend, SegmentationUnavailable, Segmenter

VIDEO = Path("room.mp4")
LOCAL = [Detection(1, 0, (0, 0, 1, 1), np.ones((1, 1), dtype=bool))]
REMOTE = [Detection(2, 0, (0, 0, 1, 1), np.ones((1, 1), dtype=bool))]


class FakeApi:
    """API segmenter returning fixed detections, or failing."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls = 0

    def segment_video(self, video_path: Path, prompt: str) -> list[Detection]:
        del video_path, prompt
        self.calls += 1
        if self.fail:
            request = httpx.Request("POST", "https://api.meta.ai/v1/responses")
            raise NotFoundError("404", response=httpx.Response(404, request=request), body=None)
        return REMOTE


@pytest.fixture(name="local_dir")
def fixture_local_dir(tmp_path: Path) -> Path:
    save_detections(LOCAL, tmp_path / "chair.npz")
    return tmp_path


def test_auto_prefers_local_results(local_dir: Path) -> None:
    api = FakeApi()
    result = Segmenter(Backend.AUTO, local_dir, api).segment(VIDEO, "chair")
    assert result[0].object_id == 1
    assert api.calls == 0


def test_auto_falls_back_to_api(local_dir: Path) -> None:
    result = Segmenter(Backend.AUTO, local_dir, FakeApi()).segment(VIDEO, "table")
    assert result[0].object_id == 2


def test_api_backend_ignores_local_results(local_dir: Path) -> None:
    result = Segmenter(Backend.API, local_dir, FakeApi()).segment(VIDEO, "chair")
    assert result[0].object_id == 2


def test_local_backend_never_calls_api(local_dir: Path) -> None:
    api = FakeApi()
    with pytest.raises(SegmentationUnavailable, match="sam3_video.sbatch"):
        Segmenter(Backend.LOCAL, local_dir, api).segment(VIDEO, "table")
    assert api.calls == 0


def test_api_failure_explains_gpu_fallback(local_dir: Path) -> None:
    with pytest.raises(SegmentationUnavailable, match="GPU"):
        Segmenter(Backend.AUTO, local_dir, FakeApi(fail=True)).segment(VIDEO, "table")


def test_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIVOT_SAM_BACKEND", "LOCAL")
    assert Segmenter.backend_from_env() is Backend.LOCAL
    monkeypatch.delenv("PIVOT_SAM_BACKEND")
    monkeypatch.delenv("PIVOT_BACKEND", raising=False)
    assert Segmenter.backend_from_env() is Backend.AUTO
    # Without an explicit choice, SAM follows the version of PIVOT.
    monkeypatch.setenv("PIVOT_BACKEND", "open")
    assert Segmenter.backend_from_env() is Backend.LOCAL
    monkeypatch.setenv("PIVOT_BACKEND", "api")
    assert Segmenter.backend_from_env() is Backend.API
