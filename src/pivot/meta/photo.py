"""Objects brought in from a photo: what they are, how big, and their picture.

"Will this armchair fit?": the visitor photographs it (in a shop, online, at a
friend's) or picks a photo. Muse Spark names it and estimates its size from
what it looks like; SAM 3.1 cuts it out of the photo, so the page shows it.
The size is an estimate, which
the visitor can correct.
"""

from __future__ import annotations

import base64
import io
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from pivot.geometry.types import BoolArray
from pivot.meta.client import MetaConfig, call_with_retry
from pivot.perception.detections import Detection

DESCRIBE_PROMPT = (
    "This photo shows a piece of furniture or another object someone may bring into their "
    "home. Name it with one short, common English noun, as one would to a mover (\"armchair\", "
    "\"bookshelf\", \"floor lamp\"), and estimate its real size in meters: width (the longer "
    "side seen from above), depth and height. Use what the photo shows to judge the scale, "
    "or the usual size of such an object. Describe its look in a few words (color, material). "
    'Answer with a JSON object only: {"label": ..., "width_m": ..., "depth_m": ..., '
    '"height_m": ..., "look": ...}; if the photo shows no such object, {"label": null}.')
# Estimates outside these bounds are clamped, in meters.
_SMALLEST, _LARGEST = 0.1, 4.0
# Nouns SAM 3.1 is asked for after the label, when the label finds nothing: SAM
# knows common nouns better than precise ones ("sideboard" found nothing in a
# photo of one).
_FALLBACK_NOUNS = ("furniture", "object")
# Notches in the top of a mask at most this share of its width wide are filled.
_WIDEST_NOTCH = 0.25


@dataclass(frozen=True)
class PhotoObject:
    """An object seen in a photo.

    Attributes:
        label: What it is.
        size: Estimated width, depth and height, in meters.
        look: Its color and material, in a few words.
        picture: The object cut out of the photo, as a PNG with a transparent
            background; the whole photo if it could not be cut out.
    """

    label: str
    size: tuple[float, float, float]
    look: str
    picture: bytes


def parse_description(text: str) -> tuple[str, tuple[float, float, float], str] | None:
    """Reads the model's description of the photo.

    Args:
        text: The model's answer.

    Returns:
        Label, size (width >= depth) and look; None if no object was named.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        data: dict[str, Any] = json.loads(match.group(0)) if match else {}
        label = " ".join(str(data.get("label") or "").lower().split())
        width, depth, height = (float(data[k]) for k in ("width_m", "depth_m", "height_m"))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
    if not label:
        return None
    width, depth = max(width, depth), min(width, depth)
    width, depth, height = (min(max(v, _SMALLEST), _LARGEST) for v in (width, depth, height))
    return label, (width, depth, height), str(data.get("look") or "")


def describe_photo(config: MetaConfig, photo: bytes, client: Any = None) -> str:
    """Asks Muse Spark what a photo shows and how big it is.

    Args:
        config: API configuration; ``vision_model`` is used.
        photo: JPEG or PNG.
        client: SDK client; created from ``config`` if None.

    Returns:
        The model's answer.
    """
    client = client if client is not None else config.client()
    kind = "png" if photo.startswith(b"\x89PNG") else "jpeg"
    url = f"data:image/{kind};base64," + base64.b64encode(photo).decode()
    effort = {"reasoning_effort": config.reasoning_effort} if config.reasoning_effort else {}
    response = call_with_retry(lambda: client.chat.completions.create(
        model=config.vision_model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": DESCRIBE_PROMPT},
            {"type": "image_url", "image_url": {"url": url}}]}],
        **effort))
    return str(response.choices[0].message.content or "")


def cut_out(photo: bytes, detections: list[Detection]) -> bytes:
    """Cuts the main detected object out of a photo.

    Args:
        photo: JPEG or PNG.
        detections: Its segmentation; the largest mask is the object.

    Returns:
        A PNG of the object's box, transparent around it; the photo as a PNG
        if nothing was detected.
    """
    image = Image.open(io.BytesIO(photo)).convert("RGBA")
    if detections:
        best = max(detections, key=lambda d: int(d.mask.sum()))
        left, top, right, bottom = best.bounds
        image = image.crop((left, top, right, bottom))
        mask = fill_top_notches(np.asarray(best.mask, dtype=bool))
        alpha = mask[:image.height, :image.width].astype(np.uint8) * 255
        canvas = np.zeros((image.height, image.width), dtype=np.uint8)
        canvas[:alpha.shape[0], :alpha.shape[1]] = alpha
        image.putalpha(Image.fromarray(canvas))
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def fill_top_notches(mask: BoolArray) -> BoolArray:
    """Fills the notches left in the top of an object by what stands on it.

    A vase on a sideboard is not part of the sideboard, so SAM leaves it out,
    and the sideboard's top shows a notch where the vase hid it. A notch is a
    narrow dip of the mask's top edge with higher edge on both sides; the gap
    under a table, between its legs, opens downward and stays, and so does the
    slope of a top seen in perspective.

    Args:
        mask: (H, W) mask of the object.

    Returns:
        The mask, its top notches filled.
    """
    height, width = mask.shape
    seen = mask.any(axis=0)
    top = np.where(seen, mask.argmax(axis=0), height)
    # The highest edge on each side bounds the fill, as water fills a basin.
    level = np.maximum(np.minimum.accumulate(top), np.minimum.accumulate(top[::-1])[::-1])
    runs, count = ndimage.label(seen & (level < top))
    filled = mask.copy()
    for run in range(1, count + 1):
        columns = np.flatnonzero(runs == run)
        if len(columns) <= _WIDEST_NOTCH * width:
            for column in columns:
                filled[level[column]:top[column], column] = True
    return filled


def object_from_photo(config: MetaConfig, photo: bytes, segment: Any = None,
                      client: Any = None) -> PhotoObject | None:
    """Names an object in a photo, estimates its size and cuts it out.

    Args:
        config: API configuration.
        photo: JPEG or PNG.
        segment: Function ``(image path, noun) -> detections`` (SAM 3.1); the
            photo is not cut out if None.
        client: SDK client of the vision model; created from ``config`` if None.

    Returns:
        The object, or None if the photo shows none.
    """
    described = parse_description(describe_photo(config, photo, client))
    if described is None:
        return None
    label, size, look = described
    detections: list[Detection] = []
    if segment is not None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / ("photo.png" if photo.startswith(b"\x89PNG") else "photo.jpg")
            path.write_bytes(photo)
            for noun in nouns_to_try(label):
                try:
                    detections = segment(path, noun)
                except Exception:  # pylint: disable=broad-exception-caught
                    # The picture is a nicety: without the cut-out, the photo itself is shown.
                    break
                if detections:
                    break
    return PhotoObject(label, size, look, cut_out(photo, detections))


def nouns_to_try(label: str) -> list[str]:
    """Lists the nouns to ask SAM 3.1 for, most precise first.

    Args:
        label: The object's name, e.g. ``"leather armchair"``.

    Returns:
        The label, its last word, then common nouns, without repeats.
    """
    return list(dict.fromkeys([label, label.split()[-1], *_FALLBACK_NOUNS]))
