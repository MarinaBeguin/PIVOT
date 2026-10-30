"""Tests for the transcription of spoken requests."""

import io
import json
import wave
from typing import Any

import numpy as np
import pytest

from pivot.geometry.types import FloatArray
from pivot.meta.client import MetaConfig
from pivot.server.speech import read_wav, transcribe_api


def wav(samples: FloatArray, rate: int = 16000, channels: int = 1) -> bytes:
    buffer = io.BytesIO()
    with wave.Wave_write(buffer) as file:
        file.setnchannels(channels)
        file.setsampwidth(2)
        file.setframerate(rate)
        file.writeframes((samples * 32767).astype("<i2").tobytes())
    return buffer.getvalue()


def test_read_wav_averages_channels_and_scales_to_unit() -> None:
    stereo = np.array([[0.5, -0.5], [0.25, 0.25]]).ravel()
    samples, rate = read_wav(wav(stereo, 24000, channels=2))
    assert rate == 24000
    assert samples == pytest.approx([0.0, 0.25], abs=1e-4)


class FakeResponse:
    """The API's answer to a transcription request."""

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return {"sessionId": "s", "transcript": " Move the chair. ", "turns": []}


def test_api_transcription_sends_multipart_with_keywords() -> None:
    calls: list[dict[str, Any]] = []

    def post(url: str, **kwargs: Any) -> FakeResponse:
        calls.append({"url": url, **kwargs})
        return FakeResponse()

    config = MetaConfig(api_key="k", base_url="https://api.meta.ai/v1/")
    assert transcribe_api(config, b"RIFF", ["chair"], post=post) == "Move the chair."
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://api.meta.ai/v1/asr/transcribe"
    assert call["headers"]["Authorization"] == "Bearer k"
    settings = json.loads(call["files"]["request"][1])
    assert settings == {"model": "muse-voice-transcribe-1.0", "audioEncoding": "WAV",
                        "mode": "PUSH_TO_TALK", "keywords": ["chair"]}
    assert call["files"]["audio"] == ("speech.wav", b"RIFF", "audio/wav")
