"""SAM 3.1 through the Meta Model API: text-prompted segmentation and tracking.

Responses are streamed as text lines (one per frame) and parsed with Meta's
official ``meta-sam-parser``. Every raw response is cached on disk, keyed by
model, prompt and media content, because video segmentation is billed per
frame and the pipeline is re-run many times during development.

The same tracking runs on our own GPU with the released weights, see
:mod:`pivot.perception.sam3_local`; both write the same detection files::

    python -m pivot.meta.sam --video room.mp4 --out detections/ --prompts chair table
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import mimetypes
from pathlib import Path
from typing import Literal

import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationMaskRecord,
    SegmentationResult,
    decode_mask_to_raster,
    image_segmentation_format,
    video_segmentation_format,
)
from openai import OpenAI

from pivot.meta.client import MetaConfig, call_with_retry
from pivot.perception.detections import Detection
from pivot.perception.sam3_local import track_prompts

Media = Literal["image", "video"]

# SAM requests fail with a spurious ``model_not_found`` far more often than
# Muse requests (about nine times out of ten on Oct 27, 2026).
_SAM_ATTEMPTS = 25


def _parse(text: str, media: Media) -> SegmentationResult:
    """Runs Meta's incremental parser over a complete response text.

    Args:
        text: Concatenated ``output_text`` of the response.
        media: Whether the request was an image or a video.

    Returns:
        The parser's final result.
    """
    if media == "image":
        image_parser = image_segmentation_format().create_parser()
        image_parser.push(text, emit=False)
        return image_parser.finish(CompletedOutcome()).result
    video_parser = video_segmentation_format().create_parser()
    video_parser.push(text, emit=False)
    return video_parser.finish(CompletedOutcome()).result


def parse_sam_output(text: str, media: Media) -> list[Detection]:
    """Parses the text output of a SAM 3.1 response.

    Args:
        text: Concatenated ``output_text`` of the response.
        media: Whether the request was an image or a video.

    Returns:
        Every detection, in output order.

    Raises:
        ValueError: If the parser reports an error.
    """
    result = _parse(text, media)
    errors = [d for d in result.diagnostics if d.severity == "error"]
    if errors:
        raise ValueError(f"Malformed SAM output: {errors[0]}")

    detections = []
    for record in result.records:
        if not isinstance(record, SegmentationMaskRecord):
            continue
        raster = np.frombuffer(decode_mask_to_raster(record.mask), dtype=np.uint8)
        mask = raster.reshape(record.mask.height, record.mask.width).astype(bool)
        bounds = record.bounds
        detections.append(Detection(
            object_id=int(record.object_id),
            frame=record.frame.frame_index if record.frame is not None else 0,
            bounds=(int(bounds.left), int(bounds.top), int(bounds.right), int(bounds.bottom)),
            mask=mask,
            confidence=record.confidence,
        ))
    return detections


def _data_url(path: Path) -> str:
    """Encodes a file as a base64 data URL.

    Args:
        path: Image file.

    Returns:
        The data URL.
    """
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


class SamClient:
    """Cached SAM 3.1 segmentation through the Meta Model API.

    Args:
        config: API configuration.
        cache_dir: Directory holding raw responses.
        client: SDK client; created from ``config`` if None.
    """

    def __init__(self, config: MetaConfig, cache_dir: Path, client: OpenAI | None = None):
        self._config = config
        self._cache_dir = cache_dir
        self._client = client if client is not None else config.client()

    def _cache_path(self, prompt: str, media_path: Path) -> Path:
        """Returns the cache file of a request.

        Args:
            prompt: Noun phrase.
            media_path: Image or video file.

        Returns:
            The cache path, keyed by model, prompt and file content.
        """
        digest = hashlib.sha256()
        digest.update(self._config.sam_model.encode())
        digest.update(b"\0" + prompt.strip().lower().encode() + b"\0")
        digest.update(media_path.read_bytes())
        return self._cache_dir / f"{digest.hexdigest()[:24]}.txt"

    def _stream_text(self, content: list[dict[str, str]]) -> str:
        """Sends one request and collects its streamed output text.

        Args:
            content: User message content parts.

        Returns:
            The concatenated output text.
        """
        chunks: list[str] = []
        # The SDK's request types do not know SAM's ``input_video`` part.
        with self._client.responses.stream(
            model=self._config.sam_model,
            input=[{"role": "user", "content": content}],  # type: ignore[misc,list-item]
            metadata={"include_confidence": "true"},
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    chunks.append(event.delta)
        return "".join(chunks)

    def _cached(self, prompt: str, media_path: Path, content: list[dict[str, str]]) -> str:
        """Returns a cached response, or requests and caches it.

        Args:
            prompt: Noun phrase.
            media_path: Image or video file.
            content: Request content, used on a cache miss.

        Returns:
            The raw output text.
        """
        path = self._cache_path(prompt, media_path)
        if path.exists():
            return path.read_text(encoding="utf-8")
        text = call_with_retry(lambda: self._stream_text(content), attempts=_SAM_ATTEMPTS)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return text

    def segment_image(self, image_path: Path, prompt: str) -> list[Detection]:
        """Segments every instance of a noun phrase in an image.

        Args:
            image_path: JPEG or PNG image.
            prompt: Short noun phrase, e.g. ``"office chair"``.

        Returns:
            The detections.
        """
        content = [{"type": "input_text", "text": prompt},
                   {"type": "input_image", "image_url": _data_url(image_path)}]
        return parse_sam_output(self._cached(prompt, image_path, content), "image")

    def segment_video(self, video_path: Path, prompt: str) -> list[Detection]:
        """Segments and tracks every instance of a noun phrase in a video.

        The video is uploaded through the Files API on a cache miss.

        Args:
            video_path: H.264 MP4 video.
            prompt: Short noun phrase.

        Returns:
            The detections of every frame.
        """
        path = self._cache_path(prompt, video_path)
        if not path.exists():
            uploaded = call_with_retry(lambda: self._client.files.create(
                file=(video_path.name, video_path.read_bytes()), purpose="user_data"))
            content = [{"type": "input_text", "text": prompt},
                       {"type": "input_video", "file_id": uploaded.id}]
            return parse_sam_output(self._cached(prompt, video_path, content), "video")
        return parse_sam_output(path.read_text(encoding="utf-8"), "video")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: tracks prompts through a video with the API.

    Writes one file per prompt in the format of :mod:`pivot.perception.sam3_local`,
    so that :mod:`pivot.build` reads either.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description="SAM 3.1 video tracking via the Meta API.")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--cache", type=Path, default=None,
                        help="Raw response cache; defaults to <out>/raw.")
    args = parser.parse_args(argv)

    client = SamClient(MetaConfig.from_env(args.env_file), args.cache or args.out / "raw")
    track_prompts(args.prompts, args.out, lambda prompt: client.segment_video(args.video, prompt))


if __name__ == "__main__":
    main()
