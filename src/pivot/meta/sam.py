"""SAM 3.1 through the Meta Model API: text-prompted segmentation and tracking.

Responses are streamed as text lines (one per frame) and parsed with Meta's
official ``meta-sam-parser``. Every raw response is cached on disk, keyed by
model, prompt and media content, because video segmentation is billed per
frame and the pipeline is re-run many times during development.

The same tracking runs on our own GPU with the released weights, see
:mod:`pivot.perception.sam3_local`; both write the same detection files.

Requests run in parallel: every kind of object at once, and a long video cut
into overlapping pieces tracked at the same time, whose tracks are joined
afterwards (see :mod:`pivot.perception.chunks`). Each video is uploaded once.
Decoding the masks of a response takes far longer than receiving it (Meta's
parser decodes about three masks a second, in pure Python): responses are
decoded in separate processes, one per core, as they arrive.
Only the frames listed in ``--frames`` (see :mod:`pivot.perception.keyframes`)
are tracked; detections keep the frame numbers of the whole video::

    python -m pivot.meta.sam --video room.mp4 --out detections/ --prompts chair table \
        --frames keyframes.json --workers 32
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import multiprocessing
import os
import threading
import time
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait,
)
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Literal

from meta_sam_parser import (
    CompletedOutcome,
    SegmentationMaskRecord,
    SegmentationResult,
    image_segmentation_format,
    video_segmentation_format,
)
from openai import OpenAI

from pivot.data.video import count_frames, select_frames, video_size
from pivot.meta import mask_codec
from pivot.meta.client import MetaConfig, call_with_retry
from pivot.perception.chunks import Linker, join_chunks, plan_chunks
from pivot.perception.cotracker import load_cotracker3, point_linker
from pivot.perception.detections import Detection
from pivot.perception.sam3_local import (
    progress_path, prompt_filename, save_detections, write_progress,
)

# Meta's parser checks masks with the compiled decoder, when numba is installed.
mask_codec.install()

Media = Literal["image", "video"]
# Tracks a noun phrase through a video file.
SegmentFunction = Callable[[Path, str], list[Detection]]
# Fetches the raw response of a noun phrase on a video file, cached on disk.
FetchFunction = Callable[[Path, str], Path | None]

# SAM requests fail with a spurious ``model_not_found`` far more often than
# Muse requests (about nine times out of ten on Oct 27, 2026).
_SAM_ATTEMPTS = 25
# Requests at once. The API allows 3,000 a minute; each one streams for minutes.
WORKERS = 32
# Processes decoding responses at once (by default, one per core up to this).
PARSE_WORKERS = 16
# Size of the pieces of a response decoded by one process at a time, in
# characters: decoding time follows the size of the masks, so a big response is
# spread over every process.
BLOCK_CHARS = 100_000
# Most frames of one piece of video (0: the whole video in one request). Off by
# default: on the office, tracks cut where an object is out of view at a cut
# changed 7 of 23 objects, as their views no longer added up (see pivot.perception.chunks).
CHUNK_FRAMES = 0


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
    with mask_codec.keep_decoded() as decoded:
        result = _parse(text, media)
        errors = [d for d in result.diagnostics if d.severity == "error"]
        if errors:
            raise ValueError(f"Malformed SAM output: {errors[0]}")

        detections = []
        for record in result.records:
            if not isinstance(record, SegmentationMaskRecord):
                continue
            # The parser decoded the mask to check it; decoded again only without numba.
            mask = decoded.get(id(record.mask))
            if mask is None:
                mask = mask_codec.decode_mask(record.mask)
            bounds = record.bounds
            detections.append(Detection(
                object_id=int(record.object_id),
                frame=record.frame.frame_index if record.frame is not None else 0,
                bounds=(int(bounds.left), int(bounds.top), int(bounds.right),
                        int(bounds.bottom)),
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
        # Requests run in threads: each file is hashed and uploaded once.
        self._lock = threading.Lock()
        self._digests: dict[Path, bytes] = {}
        self._uploads: dict[bytes, str] = {}

    def _digest(self, media_path: Path) -> bytes:
        """Hashes a file's content, once per file.

        Args:
            media_path: Image or video file.

        Returns:
            The SHA-256 digest.
        """
        with self._lock:
            if media_path not in self._digests:
                self._digests[media_path] = hashlib.sha256(media_path.read_bytes()).digest()
            return self._digests[media_path]

    def _upload(self, video_path: Path) -> str:
        """Uploads a video through the Files API, once per content.

        Args:
            video_path: The video.

        Returns:
            The uploaded file's id.
        """
        digest = self._digest(video_path)
        with self._lock:
            if digest not in self._uploads:
                uploaded = call_with_retry(lambda: self._client.files.create(
                    file=(video_path.name, video_path.read_bytes()), purpose="user_data"))
                self._uploads[digest] = uploaded.id
            return self._uploads[digest]

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
        if text:  # Nothing found is not kept: the next request asks again.
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

    def fetch_video(self, video_path: Path, prompt: str) -> Path | None:
        """Fetches the raw tracking response of a noun phrase in a video, cached on disk.

        The video is uploaded through the Files API on a cache miss, once.

        Args:
            video_path: H.264 MP4 video.
            prompt: Short noun phrase.

        Returns:
            The file holding the response, or None if nothing was found.
        """
        path = self._cache_path(prompt, video_path)
        if not path.exists():
            content = [{"type": "input_text", "text": prompt},
                       {"type": "input_video", "file_id": self._upload(video_path)}]
            self._cached(prompt, video_path, content)
        return path if path.exists() else None

    def segment_video(self, video_path: Path, prompt: str) -> list[Detection]:
        """Segments and tracks every instance of a noun phrase in a video.

        Args:
            video_path: H.264 MP4 video.
            prompt: Short noun phrase.

        Returns:
            The detections of every frame.
        """
        return parse_video_response(self.fetch_video(video_path, prompt))


def parse_video_response(path: Path | None) -> list[Detection]:
    """Decodes a cached video response.

    Args:
        path: The raw response file; None when nothing was found.

    Returns:
        The detections of every frame.
    """
    return [] if path is None else parse_sam_output(path.read_text(encoding="utf-8"), "video")


def response_blocks(path: Path | None, size: int = BLOCK_CHARS) -> list[str]:
    """Cuts a video response into blocks of whole frames, decoded apart.

    A response holds one line per frame, and each frame decodes on its own.

    Args:
        path: The raw response file; None when nothing was found.
        size: Characters per block, about (a frame is never cut).

    Returns:
        The blocks, in order.
    """
    if path is None:
        return []
    blocks: list[str] = []
    current: list[str] = []
    length = 0
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        current.append(line)
        length += len(line)
        if length >= size:
            blocks.append("".join(current))
            current, length = [], 0
    return blocks + (["".join(current)] if current else [])


def parse_video_block(text: str) -> list[Detection]:
    """Decodes a block of a video response (in a worker process, see :func:`track_in_parallel`).

    Args:
        text: Whole lines of a response.

    Returns:
        Their detections.
    """
    return parse_sam_output(text, "video")


def piece_videos(video: Path, chunks: list[list[int]], directory: Path) -> list[Path]:
    """Writes the video of each piece, reused across runs.

    Args:
        video: The whole video.
        chunks: Frame indices of each piece.
        directory: Where the pieces are kept.

    Returns:
        One video per piece; the whole video itself when one piece holds every frame.
    """
    if len(chunks) == 1 and chunks[0] == list(range(count_frames(video))):
        return [video]
    paths = []
    for frames in chunks:
        key = hashlib.sha256(f"{video.resolve()}:{video.stat().st_size}:{frames}".encode())
        path = directory / f"{video.stem}_{key.hexdigest()[:12]}.mp4"
        # A piece written by an ffmpeg that repeated frames is made again: its frames
        # would not be those of the whole video.
        if not path.exists() or count_frames(path) != len(frames):
            select_frames(video, frames, path)
        paths.append(path)
    return paths


@dataclass(frozen=True)
class Decoding:
    """Responses received by threads and decoded by processes.

    Attributes:
        fetch: Receives the raw response of a prompt on a piece.
        workers: Processes decoding responses; one per core (up to
            :data:`PARSE_WORKERS`) if 0.
    """

    fetch: FetchFunction
    workers: int = 0


# pylint: disable-next=too-many-locals
def track_in_parallel(prompts: list[str], out: Path, segment: SegmentFunction,
                      pieces: list[Path], chunks: list[list[int]], workers: int,
                      linker: Linker | None = None, decoding: Decoding | None = None) -> None:
    """Tracks every prompt on every piece at once, then joins each prompt's pieces.

    Writes one detection file per prompt (skipping those already written) and,
    while a prompt runs, its progress file (the share of its pieces done).
    With ``decoding``, threads only receive the responses, and processes
    decode them as they arrive: decoding is what takes time.

    Args:
        prompts: Noun phrases.
        out: Output directory.
        segment: Tracks a prompt through one piece's video (used without ``decoding``).
        pieces: Videos of the pieces.
        chunks: Frame indices of each piece in the whole video.
        workers: Requests at once.
        linker: Joins tracks across cuts that no shared frame shows.
        decoding: Receiving and decoding apart, if given.
    """
    todo = [p for p in prompts if not (out / prompt_filename(p)).exists()]
    for prompt in prompts:
        if prompt not in todo:
            print(f"{prompt!r}: cached at {out / prompt_filename(prompt)}", flush=True)
    start = time.time()
    done: dict[str, int] = dict.fromkeys(todo, 0)
    for prompt in todo:
        write_progress(progress_path(out, prompt), 0.0)
    jobs = [(prompt, index) for prompt in todo for index in range(len(pieces))]
    results: dict[tuple[str, int], list[Detection]] = {}

    def finish(prompt: str, index: int, detections: list[Detection]) -> None:
        """Keeps one piece's detections; saves the prompt once all its pieces are in.

        Args:
            prompt: Noun phrase.
            index: Piece index.
            detections: Its detections.
        """
        results[prompt, index] = detections
        done[prompt] += 1
        write_progress(progress_path(out, prompt), done[prompt] / (len(pieces) + 1))
        if done[prompt] < len(pieces):
            return
        joined = join_chunks([results[prompt, i] for i in range(len(pieces))], chunks,
                             linker=linker)
        save_detections(joined, out / prompt_filename(prompt))
        progress_path(out, prompt).unlink(missing_ok=True)
        print(f"{prompt!r}: {len({d.object_id for d in joined})} objects, "
              f"{len(joined)} detections on {len({d.frame for d in joined})} "
              f"frames, {time.time() - start:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        if decoding is None:
            segmented = {pool.submit(segment, pieces[i], p): (p, i) for p, i in jobs}
            for future in as_completed(segmented):
                finish(*segmented[future], future.result())
            return
        # Spawned, not forked: the parent holds threads and possibly CoTracker3.
        with ProcessPoolExecutor(max_workers=decoding.workers or min(PARSE_WORKERS,
                                                                     os.cpu_count() or 1),
                                 mp_context=multiprocessing.get_context("spawn")) as parsers:
            fetched: dict[Future[Any], tuple[str, int]] = {
                pool.submit(decoding.fetch, pieces[i], p): (p, i) for p, i in jobs}
            # Each response is decoded in blocks: the job and position of each block,
            # and the blocks of each job, filled as they come back.
            decoded: dict[Future[Any], tuple[tuple[str, int], int]] = {}
            blocks: dict[tuple[str, int], list[list[Detection] | None]] = {}
            pending: set[Future[Any]] = set(fetched)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for job in finished:
                    if job in fetched:
                        texts = response_blocks(job.result())
                        blocks[fetched[job]] = [None] * len(texts)
                        if not texts:
                            finish(*fetched[job], [])
                        for position, text in enumerate(texts):
                            parsed = parsers.submit(parse_video_block, text)
                            decoded[parsed] = (fetched[job], position)
                            pending.add(parsed)
                        continue
                    key, position = decoded[job]
                    blocks[key][position] = job.result()
                    if all(block is not None for block in blocks[key]):
                        finish(*key, [d for block in blocks[key] for d in block or []])


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
    parser.add_argument("--frames", type=Path, default=None,
                        help="JSON list of the frames to track; every frame if omitted.")
    parser.add_argument("--workers", type=int,
                        default=int(os.environ.get("PIVOT_SAM_WORKERS", WORKERS)))
    parser.add_argument("--parse-workers", type=int,
                        default=int(os.environ.get("PIVOT_PARSE_WORKERS", "0")),
                        help="Processes decoding responses; one per core if 0.")
    parser.add_argument("--chunk-frames", type=int, default=CHUNK_FRAMES,
                        help="Most frames of one piece of video; 0 tracks it whole.")
    parser.add_argument("--no-cotracker", action="store_true",
                        help="Join pieces by mask overlap only.")
    args = parser.parse_args(argv)

    total = count_frames(args.video)
    frames = (json.loads(args.frames.read_text(encoding="utf-8")) if args.frames
              else list(range(total)))
    chunks = plan_chunks(frames, args.chunk_frames) if args.chunk_frames > 0 else [frames]
    cache = args.cache or args.out / "raw"
    linker = None if len(chunks) == 1 or args.no_cotracker else _cotracker_linker(args.video)
    print(f"{len(frames)} of {total} frames in {len(chunks)} piece(s)"
          + (", joined with CoTracker3" if linker else ""), flush=True)
    client = SamClient(MetaConfig.from_env(args.env_file), cache)
    track_in_parallel(args.prompts, args.out, client.segment_video,
                      piece_videos(args.video, chunks, cache / "pieces"), chunks,
                      args.workers, linker, Decoding(client.fetch_video, args.parse_workers))


def _cotracker_linker(video: Path) -> Linker | None:
    """Builds the CoTracker3 linker of a video, if CoTracker3 can be loaded.

    Args:
        video: The whole video.

    Returns:
        The linker, or None (tracks are then joined by mask overlap only).
    """
    tracker = load_cotracker3()
    return point_linker(video, video_size(video), tracker) if tracker else None


if __name__ == "__main__":
    main()
