"""Speech to text for spoken requests (a microphone instead of a keyboard, as on glasses).

The page records a short WAV (mono, 16-bit, 16 kHz) and the server turns it
into text, which then goes to the agent like a typed request. Two backends,
like the rest of PIVOT: Muse Voice Transcribe through the Meta API, or Meta's
wav2vec 2.0 (English) with the released weights, on the CPU.
"""

from __future__ import annotations

import io
import json
import os
import wave
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np

from pivot.geometry.types import FloatArray
from pivot.meta.client import MetaConfig

# From a WAV recording and words to favor (the scene's labels) to text.
Transcriber = Callable[[bytes, list[str]], str]

API_MODEL = "muse-voice-transcribe-1.0"
OPEN_MODEL = "facebook/wav2vec2-large-960h-lv60-self"


def read_wav(data: bytes) -> tuple[FloatArray, int]:
    """Decodes a 16-bit PCM WAV, averaging channels.

    Args:
        data: The WAV file's bytes.

    Returns:
        A tuple ``(samples, rate)``: samples in [-1, 1] and the sample rate in Hz.

    Raises:
        ValueError: If the file is not 16-bit PCM.
    """
    with wave.open(io.BytesIO(data)) as file:
        if file.getsampwidth() != 2:
            raise ValueError("Only 16-bit PCM WAV is supported.")
        frames = np.frombuffer(file.readframes(file.getnframes()), dtype="<i2")
        samples = frames.reshape(-1, file.getnchannels()).mean(axis=1) / 32768.0
        return samples.astype(np.float64), file.getframerate()


def transcribe_api(config: MetaConfig, audio: bytes, keywords: list[str] | None = None,
                   post: Callable[..., Any] = httpx.post) -> str:
    """Transcribes a WAV with Muse Voice Transcribe.

    Args:
        config: API configuration (key and base URL).
        audio: WAV recording, mono 16-bit PCM at 16 or 24 kHz.
        keywords: Words to favor, such as the labels of the scene's objects.
        post: HTTP POST function, injectable for tests.

    Returns:
        The transcript.

    Raises:
        httpx.HTTPStatusError: If the API refuses the request.
    """
    settings: dict[str, Any] = {"model": API_MODEL, "audioEncoding": "WAV",
                                "mode": "PUSH_TO_TALK"}
    if keywords:
        settings["keywords"] = keywords
    response = post(
        f"{config.base_url.rstrip('/')}/asr/transcribe",
        headers={"Authorization": f"Bearer {config.api_key}", "Accept": "application/json"},
        files={"request": (None, json.dumps(settings), "application/json"),
               "audio": ("speech.wav", audio, "audio/wav")},
        timeout=60.0)
    response.raise_for_status()
    return str(response.json()["transcript"]).strip()


class OpenTranscriber:
    """wav2vec 2.0 with the released weights, on the CPU; loaded on first use.

    Args:
        model: Hugging Face model id.
    """

    def __init__(self, model: str = OPEN_MODEL):
        self._name = model
        self._model: Any = None
        self._processor: Any = None

    def __call__(self, audio: bytes, keywords: list[str] | None = None) -> str:
        """Transcribes a WAV recording.

        Args:
            audio: 16-bit PCM WAV.
            keywords: Ignored: this model takes no vocabulary hints.

        Returns:
            The transcript, lowercase (the model writes no punctuation).
        """
        # Optional dependencies of the open version, installed on the cluster.
        # pylint: disable=import-outside-toplevel,import-error
        import torch
        from transformers import (  # type: ignore[import-not-found]
            Wav2Vec2ForCTC,
            Wav2Vec2Processor,
        )

        # pylint: enable=import-outside-toplevel,import-error
        if self._model is None:
            self._processor = Wav2Vec2Processor.from_pretrained(self._name)
            self._model = Wav2Vec2ForCTC.from_pretrained(self._name).eval()
        samples, rate = read_wav(audio)
        if rate != 16000:
            # The page records at 16 kHz; linear resampling covers other files.
            times = np.arange(0.0, len(samples) / rate, 1.0 / 16000)
            samples = np.interp(times, np.arange(len(samples)) / rate, samples)
        inputs = self._processor(samples.astype(np.float32), sampling_rate=16000,
                                 return_tensors="pt")
        with torch.no_grad():
            logits = self._model(inputs.input_values).logits
        return str(self._processor.batch_decode(logits.argmax(dim=-1))[0]).lower().strip()


def transcriber_from_env(config: MetaConfig) -> Transcriber:
    """Chooses the transcription backend from ``PIVOT_BACKEND``.

    Args:
        config: API configuration; ignored by the open version.

    Returns:
        A function from a WAV recording and words to favor to text.
    """
    if os.environ.get("PIVOT_BACKEND", "api").lower() == "open":
        return OpenTranscriber()
    return lambda audio, keywords: transcribe_api(config, audio, keywords)
