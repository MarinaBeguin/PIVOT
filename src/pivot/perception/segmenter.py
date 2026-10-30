"""One entry point for video segmentation, whichever SAM 3.1 backend serves it.

Two backends produce identical :class:`Detection` lists:

* ``api``: SAM 3.1 through the Meta Model API (:mod:`pivot.meta.sam`).
* ``local``: results of a self-hosted SAM 3.1 GPU job
  (:mod:`pivot.perception.sam3_local`), read from its output directory.

``auto`` prefers local results when they exist and falls back to the API.
The backend is chosen with ``PIVOT_SAM_BACKEND``; by default it follows
``PIVOT_BACKEND`` (``api`` uses the API, ``open`` the self-hosted results).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from openai import APIError

from pivot.perception.detections import Detection
from pivot.perception.sam3_local import load_detections, prompt_filename


class Backend(str, Enum):
    """Available segmentation backends."""

    AUTO = "auto"
    API = "api"
    LOCAL = "local"


class VideoSegmenter(Protocol):
    """Anything that segments a video from a text prompt."""

    def segment_video(self, video_path: Path, prompt: str) -> list[Detection]:
        """Segments and tracks a noun phrase through a video.

        Args:
            video_path: H.264 MP4 video.
            prompt: Noun phrase.

        Returns:
            The detections of every frame.
        """


class SegmentationUnavailable(RuntimeError):
    """No backend could provide detections for a prompt."""


@dataclass
class Segmenter:
    """Dispatches segmentation requests to the configured backend.

    Attributes:
        backend: Backend selection policy.
        local_dir: Output directory of the self-hosted GPU job for this video.
        api: Meta Model API segmenter, or None if no API access is configured.
    """

    backend: Backend
    local_dir: Path
    api: VideoSegmenter | None = None

    @staticmethod
    def backend_from_env() -> Backend:
        """Reads the backend policy from ``PIVOT_SAM_BACKEND``, else ``PIVOT_BACKEND``.

        Returns:
            The configured backend; ``auto`` if neither variable is set.
        """
        explicit = os.environ.get("PIVOT_SAM_BACKEND")
        if explicit:
            return Backend(explicit.lower())
        version = os.environ.get("PIVOT_BACKEND", "").lower()
        return {"api": Backend.API, "open": Backend.LOCAL}.get(version, Backend.AUTO)

    def segment(self, video_path: Path, prompt: str) -> list[Detection]:
        """Returns the detections of a prompt in a video.

        Args:
            video_path: The video, as sent to SAM 3.1.
            prompt: Noun phrase.

        Returns:
            The detections.

        Raises:
            SegmentationUnavailable: If the selected backends cannot serve the
                request; the message says how to produce the missing results.
        """
        local_file = self.local_dir / prompt_filename(prompt)
        if self.backend is not Backend.API and local_file.exists():
            return load_detections(local_file)
        if self.backend is Backend.LOCAL:
            raise SegmentationUnavailable(
                f"No local SAM 3.1 results for {prompt!r} in {self.local_dir}; run "
                "scripts/cluster/sam3_video.sbatch.")
        if self.api is None:
            raise SegmentationUnavailable("No Meta Model API access configured.")
        try:
            return self.api.segment_video(video_path, prompt)
        except APIError as error:
            raise SegmentationUnavailable(
                f"SAM 3.1 API failed for {prompt!r} ({error}); run "
                "scripts/cluster/sam3_video.sbatch to segment on a GPU instead.") from error
