"""Answers read aloud by Meta's MMS-TTS, the same voice on every device.

The page reads answers aloud for people who cannot look at a screen (on
glasses). The browser's own voice differs from device to device and from
language to language; Meta's Massively Multilingual Speech text-to-speech
models (MMS-TTS, small VITS models on the CPU) give one voice per language:
English, French, German and Spanish. Italian and Chinese have no MMS-TTS
model: the page keeps the browser's voice for them.

The MMS-TTS weights are licensed CC-BY-NC 4.0 (non-commercial): fine for a
demo or research, not for a paid service; they are downloaded from Hugging
Face on first use, never stored in this repository. Set ``PIVOT_TTS=mms``
to enable it; it needs ``torch`` and ``transformers`` (``pip install -e
".[guard]"``).
"""

from __future__ import annotations

import importlib.util
import io
import os
import sys
import threading
import wave
from collections.abc import Callable
from typing import Any

import numpy as np

from pivot.geometry.types import FloatArray

# Page language to MMS-TTS model.
MMS_VOICES = {"en": "facebook/mms-tts-eng", "fr": "facebook/mms-tts-fra",
              "de": "facebook/mms-tts-deu", "es": "facebook/mms-tts-spa"}
# Longest text read aloud at once, in characters.
MAX_TEXT = 1500

# Reads a text aloud in a language: WAV bytes, or None if the language has no voice.
Speaker = Callable[[str, str], bytes | None]


def wav_bytes(samples: FloatArray, rate: int) -> bytes:
    """Encodes a waveform as 16-bit PCM WAV.

    Args:
        samples: Mono samples in [-1, 1].
        rate: Sampling rate, in Hz.

    Returns:
        The WAV file's bytes.
    """
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.Wave_write(buffer) as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()


class MmsSpeaker:
    """MMS-TTS voices, each loaded on first use and kept.

    Args:
        voices: Page language to Hugging Face model id.
    """

    def __init__(self, voices: dict[str, str] | None = None):
        self._voices = voices or MMS_VOICES
        self._loaded: dict[str, tuple[Any, Any]] = {}
        self._lock = threading.Lock()

    def __call__(self, text: str, language: str) -> bytes | None:
        """Reads a text aloud.

        Args:
            text: What to say.
            language: Code of the page's language.

        Returns:
            WAV bytes, or None if the language has no MMS-TTS voice.
        """
        name = self._voices.get(language)
        if name is None:
            return None
        # Optional dependencies, installed where the voice is enabled.
        # pylint: disable=import-outside-toplevel,import-error
        import torch
        from transformers import AutoTokenizer, VitsModel

        # pylint: enable=import-outside-toplevel,import-error
        with self._lock:
            if language not in self._loaded:
                self._loaded[language] = (AutoTokenizer.from_pretrained(name),
                                          VitsModel.from_pretrained(name).eval())
            tokenizer, model = self._loaded[language]
            inputs = tokenizer(text[:MAX_TEXT], return_tensors="pt")
            with torch.no_grad():
                waveform = model(**inputs).waveform[0].numpy().astype(np.float64)
        return wav_bytes(waveform, int(model.config.sampling_rate))


def speaker_from_env() -> Speaker | None:
    """Chooses the voice of the answers from ``PIVOT_TTS``.

    Returns:
        MMS-TTS with ``PIVOT_TTS=mms``, if torch and transformers are installed;
        None (the browser's voice) otherwise.
    """
    if os.environ.get("PIVOT_TTS", "").lower() != "mms":
        return None
    if any(importlib.util.find_spec(name) is None for name in ("torch", "transformers")):
        print("PIVOT_TTS=mms needs torch and transformers (pip install -e \".[guard]\"): "
              "answers are read by the browser's voice.", file=sys.stderr)
        return None
    return MmsSpeaker()
