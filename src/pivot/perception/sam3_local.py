"""Self-hosted SAM 3.1 video tracking, for GPU jobs.

The Meta Model API serves the same model; this backend runs the open
checkpoints (``facebook/sam3.1`` on Hugging Face) on our own GPU when the API
is unavailable or a video is too long. Results are stored in a compact
``.npz`` file per prompt and loaded back as :class:`Detection` objects, so the
rest of the pipeline is backend-agnostic.

Run on a GPU node::

    python -m pivot.perception.sam3_local --video room.mp4 --out detections/ \\
        --prompts chair table monitor
"""

from __future__ import annotations

import argparse
import functools
import inspect
import re
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import numpy as np

from pivot.geometry.types import BoolArray
from pivot.perception.detections import Detection


def save_detections(detections: list[Detection], path: Path) -> None:
    """Writes detections to a compressed ``.npz`` file.

    Box-local masks are bit-packed and concatenated.

    Args:
        detections: Detections to store.
        path: Destination file.
    """
    packed = [np.packbits(d.mask.ravel()) for d in detections]
    offsets = np.cumsum([0, *(len(p) for p in packed)])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        object_id=np.array([d.object_id for d in detections], dtype=np.int64),
        frame=np.array([d.frame for d in detections], dtype=np.int64),
        bounds=np.array([d.bounds for d in detections], dtype=np.int64).reshape(-1, 4),
        confidence=np.array([np.nan if d.confidence is None else d.confidence
                             for d in detections], dtype=np.float64),
        offsets=offsets.astype(np.int64),
        bits=np.concatenate(packed) if packed else np.zeros(0, dtype=np.uint8),
    )


def load_detections(path: Path) -> list[Detection]:
    """Reads detections written by :func:`save_detections`.

    Args:
        path: Source file.

    Returns:
        The detections.
    """
    data = np.load(path)
    detections = []
    for i, (left, top, right, bottom) in enumerate(data["bounds"].tolist()):
        shape = (bottom - top, right - left)
        bits = data["bits"][data["offsets"][i]:data["offsets"][i + 1]]
        mask = np.unpackbits(bits, count=shape[0] * shape[1]).reshape(shape).astype(bool)
        confidence = float(data["confidence"][i])
        detections.append(Detection(
            int(data["object_id"][i]), int(data["frame"][i]), (left, top, right, bottom), mask,
            None if np.isnan(confidence) else confidence))
    return detections


def detection_from_mask(object_id: int, frame: int, mask: BoolArray,
                        confidence: float | None) -> Detection | None:
    """Crops a full-frame mask to its bounding box.

    Args:
        object_id: Object identifier.
        frame: Frame index.
        mask: (H, W) full-frame mask.
        confidence: Detection confidence.

    Returns:
        The detection, or None if the mask is empty.
    """
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None
    top, bottom, left, right = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    return Detection(object_id, frame, (int(left), int(top), int(right), int(bottom)),
                     mask[top:bottom, left:right].copy(), confidence)


def prompt_filename(prompt: str) -> str:
    """Turns a prompt into a safe file name.

    Args:
        prompt: Noun phrase.

    Returns:
        A lowercase, underscore-separated ``.npz`` name.
    """
    return re.sub(r"[^a-z0-9]+", "_", prompt.strip().lower()).strip("_") + ".npz"


def progress_path(out: Path, prompt: str) -> Path:
    """Returns the file reporting how far the tracking of a prompt is.

    It holds a permille (0-1000), so that the shell scripts showing a progress
    bar need integer arithmetic only. It is removed once the prompt is done.

    Args:
        out: Output directory.
        prompt: Noun phrase.

    Returns:
        The path next to the prompt's detection file.
    """
    return (out / prompt_filename(prompt)).with_suffix(".progress")


def write_progress(path: Path, fraction: float) -> None:
    """Writes a progress file atomically, so that readers never see it half written.

    Args:
        path: Progress file.
        fraction: Fraction done, in [0, 1].
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".progress.tmp")
    partial.write_text(str(round(1000 * min(max(fraction, 0.0), 1.0))), encoding="utf-8")
    partial.replace(path)


def drop_unsupported_kwargs(function: Callable[..., Any]) -> Callable[..., Any]:
    """Wraps a function so that keyword arguments it does not declare are ignored.

    ``Sam3BasePredictor.start_session`` always passes ``offload_state_to_cpu``
    to ``init_state``, which the SAM 3.1 multiplex model does not accept
    (sam3 commit f1647e8). Filtering by signature keeps working once upstream
    fixes either side.

    Args:
        function: The function to wrap.

    Returns:
        The wrapped function.
    """
    accepted = set(inspect.signature(function).parameters)

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Calls ``function`` with the declared keyword arguments only.

        Args:
            *args: Positional arguments, passed through.
            **kwargs: Keyword arguments, filtered.

        Returns:
            What ``function`` returns.
        """
        return function(*args, **{k: v for k, v in kwargs.items() if k in accepted})

    return wrapper


def track_prompt(predictor: Any, session_id: str, prompt: str,
                 on_frame: Callable[[int], None] = lambda frame: None) -> list[Detection]:
    """Detects and tracks every instance of a prompt through a loaded video.

    Args:
        predictor: A SAM 3.1 multiplex video predictor.
        session_id: Session holding the loaded video.
        prompt: Noun phrase.
        on_frame: Called with the index of each frame once it is processed.

    Returns:
        Detections of every frame.
    """
    # A session must be reset before switching to another text prompt.
    predictor.handle_request(request={"type": "reset_session", "session_id": session_id})
    predictor.handle_request(request={"type": "add_prompt", "session_id": session_id,
                                      "frame_index": 0, "text": prompt})
    detections = []
    for response in predictor.handle_stream_request(
            # The prompt is on frame 0, so a backward pass adds nothing. Outputs
            # are consumed right away: evicting them keeps GPU memory flat
            # instead of growing with every frame.
            request={"type": "propagate_in_video", "session_id": session_id,
                     "propagation_direction": "forward",
                     "evict_cached_frame_outputs": True}):
        frame, outputs = int(response["frame_index"]), response["outputs"]
        probs = outputs.get("out_probs")
        for i, object_id in enumerate(outputs["out_obj_ids"].tolist()):
            confidence = float(probs[i]) if probs is not None else None
            detection = detection_from_mask(int(object_id), frame,
                                            np.asarray(outputs["out_binary_masks"][i], dtype=bool),
                                            confidence)
            if detection is not None:
                detections.append(detection)
        on_frame(frame)
    return detections


def track_prompts(prompts: list[str], out: Path, track: Callable[[str], list[Detection]],
                  extra: Callable[[], str] = lambda: "") -> None:
    """Tracks prompts one by one, writing one detection file each, skipping existing ones.

    Shared by the self-hosted model and the API client (:mod:`pivot.meta.sam`). A
    prompt's progress file exists while it runs (see :func:`progress_path`).

    Args:
        prompts: Noun phrases.
        out: Output directory.
        track: Tracks one prompt through the video.
        extra: Additional text for the progress line of a prompt.
    """
    for prompt in prompts:
        output = out / prompt_filename(prompt)
        if output.exists():
            print(f"{prompt!r}: cached at {output}", flush=True)
            continue
        start = time.time()
        write_progress(progress_path(out, prompt), 0.0)
        detections = track(prompt)
        save_detections(detections, output)
        progress_path(out, prompt).unlink(missing_ok=True)
        objects = len({d.object_id for d in detections})
        frames = len({d.frame for d in detections})
        print(f"{prompt!r}: {objects} objects, {len(detections)} detections on {frames} frames, "
              f"{time.time() - start:.0f}s{extra()}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point; requires a CUDA GPU and the ``sam3`` package.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    # Lost objects keep their tracking slot (they may reappear), so a room scan
    # fills the builder's default of 16 slots within seconds.
    parser.add_argument("--max-objects", type=int, default=64,
                        help="Tracking slots per prompt (multiple of 16).")
    args = parser.parse_args(argv)

    # Heavy GPU dependencies are only installed on the cluster.
    # pylint: disable-next=import-outside-toplevel,import-error
    import cv2  # type: ignore[import-not-found]
    # pylint: disable-next=import-outside-toplevel,import-error
    import torch
    from sam3.model_builder import (  # pylint: disable=import-outside-toplevel,import-error
        build_sam3_multiplex_video_predictor,
    )

    start = time.time()
    # FlashAttention 3 needs a Hopper GPU; our nodes are Ampere (A6000).
    predictor = build_sam3_multiplex_video_predictor(max_num_objects=args.max_objects,
                                                     use_fa3=False)
    predictor.model.init_state = drop_unsupported_kwargs(predictor.model.init_state)
    # Decoded frames (~12 MB each at 1008 px) stay in host memory.
    session = predictor.handle_request(request={"type": "start_session",
                                                "resource_path": str(args.video),
                                                "offload_video_to_cpu": True})
    print(f"Model and video loaded in {time.time() - start:.0f}s", flush=True)
    frames = max(int(cv2.VideoCapture(str(args.video)).get(cv2.CAP_PROP_FRAME_COUNT)), 1)

    def track(prompt: str) -> list[Detection]:
        """Tracks one prompt, measuring the GPU memory it takes.

        Args:
            prompt: Noun phrase.

        Returns:
            The detections.
        """
        torch.cuda.reset_peak_memory_stats()
        progress = progress_path(args.out, prompt)

        def on_frame(frame: int) -> None:
            """Reports progress every few frames, for the progress bar of the scripts.

            Args:
                frame: Index of the frame just processed.
            """
            if frame % 10 == 0:
                write_progress(progress, frame / frames)

        return track_prompt(predictor, session["session_id"], prompt, on_frame)

    track_prompts(args.prompts, args.out, track, lambda: (
        f", peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"))


if __name__ == "__main__":
    main()
