"""Blurring what is private in a user's video before anything else sees it.

A video filmed at home or in a staircase shows more than furniture: faces,
names on letter boxes and door bells, notices, parcel labels, the plate of a
car through a window. Before objects are listed, segmented or lifted, SAM 3.1
tracks these through the capture's frames (a text prompt each, as for
furniture), and every pixel it finds is blurred in the frames and the video,
with a margin. The cloud, the viewer and the moving plan then never show
them. Meta's EgoBlur (faces and plates, Apache 2.0, weights behind a form)
would do the same without the API.

    python -m pivot.privacy --capture capture/

The capture is a directory in the layout of :mod:`pivot.data.scannetpp`, with
its ``video.mp4`` (see :mod:`pivot.reconstruction.video_capture`).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

from pivot.geometry.types import BoolArray
from pivot.perception.detections import Detection

# What SAM 3.1 is asked to find: people, and writing that names or locates someone.
PROMPTS = ("person", "face", "license plate", "text", "name plate", "letter box")
# Blur strength and the margin around what is found, in pixels.
BLUR_RADIUS = 14
MARGIN = 8

Segment = Callable[[Path, str], list[Detection]]


def private_masks(video: Path, size: tuple[int, int], segment: Segment,
                  prompts: tuple[str, ...] = PROMPTS) -> dict[int, BoolArray]:
    """Finds the private pixels of every frame of a video.

    Args:
        video: The capture's video; frame ``i`` is image ``i``.
        size: ``(width, height)`` of the frames.
        segment: Tracks a text prompt through a video (SAM 3.1).
        prompts: What to look for.

    Returns:
        A full-frame mask per frame index, for the frames where something was found.
    """
    width, height = size
    masks: dict[int, BoolArray] = defaultdict(lambda: np.zeros((height, width), dtype=bool))
    for prompt in prompts:
        for detection in segment(video, prompt):
            masks[detection.frame] |= detection.full_mask(width, height)
    return dict(masks)


def blur(image: Image.Image, mask: BoolArray, radius: int = BLUR_RADIUS,
         margin: int = MARGIN) -> Image.Image:
    """Blurs an image where a mask is set, with a margin around it.

    Args:
        image: The frame.
        mask: (height, width) pixels to hide.
        radius: Gaussian blur radius, in pixels.
        margin: Growth of the mask, in pixels.

    Returns:
        The frame, blurred there.
    """
    grown = ndimage.binary_dilation(mask, iterations=margin) if margin else mask
    hidden = image.filter(ImageFilter.GaussianBlur(radius))
    return Image.composite(hidden, image, Image.fromarray(grown.astype(np.uint8) * 255))


def blur_capture(capture: Path, segment: Segment, prompts: tuple[str, ...] = PROMPTS) -> int:
    """Blurs the private parts of a capture's frames, and records what was done.

    Args:
        capture: The capture's directory.
        segment: Tracks a text prompt through a video (SAM 3.1).
        prompts: What to look for.

    Returns:
        The number of frames blurred.
    """
    frames = sorted((capture / "iphone" / "rgb").glob("frame_*.jpg"))
    with Image.open(frames[0]) as first:
        size = first.size
    masks = private_masks(capture / "video.mp4", size, segment, prompts)
    for index, mask in masks.items():
        if index < len(frames) and mask.any():
            with Image.open(frames[index]) as image:
                blurred = blur(image.convert("RGB"), mask)
            blurred.save(frames[index], quality=92)
    (capture / "privacy.json").write_text(json.dumps(
        {"prompts": list(prompts), "frames_blurred": sorted(masks)}), encoding="utf-8")
    return len(masks)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: blurs a capture through the Meta API, then re-encodes its video.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    # pylint: disable=import-outside-toplevel
    from pivot.meta.client import MetaConfig
    from pivot.meta.sam import SamClient
    from pivot.reconstruction.video_capture import capture_video

    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    args = parser.parse_args(argv)
    if (args.capture / "privacy.json").exists():
        print(f"already blurred: {args.capture}")
        return
    client = SamClient(MetaConfig.from_env(args.env_file), args.capture / "privacy_raw")
    count = blur_capture(args.capture, client.segment_video, tuple(args.prompts))
    # The video SAM 3.1 tracks furniture in is made again, from the blurred frames.
    capture_video(args.capture)
    print(f"{count} frames blurred ({', '.join(args.prompts)}) -> {args.capture}")


if __name__ == "__main__":
    main()
