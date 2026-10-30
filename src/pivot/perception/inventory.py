"""Object inventory: which kinds of objects a room video shows.

SAM 3.1 tracks the objects named by a text prompt but does not choose the names.
A vision-language model looks at frames spread over the video, tiled on one
contact sheet, and names the kinds of objects that matter for rearranging the
room; SAM 3.1 then tracks each. Two backends, like the rest of PIVOT: Muse
Spark through the Meta API, or Llama 3.2 Vision with the released weights on
our own GPU::

    python -m pivot.perception.inventory --video room.mp4 --out prompts.txt [--backend open]

Writes one prompt per line, and the contact sheet next to it (``.jpg``).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from pivot.geometry.types import UInt8Array
from pivot.meta.client import MetaConfig, call_with_retry

INVENTORY_PROMPT = (
    "These images are frames of one video walking through a room. List the kinds of objects "
    "that matter for rearranging its furniture: furniture and other objects a person could move "
    "(chairs, tables, lamps, bins...), and the doors and windows, which constrain where "
    "furniture can go. Ignore walls, floor, ceiling, small items lying on surfaces (cups, "
    "papers, keyboards) and anything outside the room. Use short, common English nouns in the "
    'singular, as one would name them to a mover ("chair", "table", "trash can"); name each kind '
    'once even if there are several, and write "door" and "window" exactly so. List at most '
    "{limit} kinds, the largest objects first. Answer with a JSON array of strings only.")
OPEN_VISION_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
# Each prompt costs SAM 3.1 a pass over the whole video (minutes); the run
# scripts allow more through the API, which is faster than our GPUs.
MAX_PROMPTS = 12
_ARTICLE = re.compile(r"^(an?|the)\s+")


def parse_inventory(text: str, limit: int = MAX_PROMPTS) -> list[str]:
    """Turns the model's answer into distinct SAM prompts.

    Reads the first JSON array of the answer, or else one name per line or
    comma. Names are lowercased, without article; a name ending with another
    listed name ("office chair", "chair") is dropped, as SAM would track the
    same objects twice.

    Args:
        text: The model's answer.
        limit: Maximum number of prompts kept, in the model's order.

    Returns:
        The prompts.
    """
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    try:
        items = [str(item) for item in json.loads(match.group(0))] if match else []
    except json.JSONDecodeError:
        items = []
    if not items:
        items = re.split(r"[,\n]", text)
    names: list[str] = []
    for item in items:
        name = " ".join(re.sub(r"[^a-z ]+", " ", item.lower()).split())
        name = _ARTICLE.sub("", name)
        if name and len(name.split()) <= 3 and name not in names:
            names.append(name)
    generic = [name for name in names
               if not any(name != other and name.endswith(" " + other) for other in names)]
    return generic[:limit]


def contact_sheet(tiles: list[UInt8Array], columns: int = 3) -> UInt8Array:
    """Tiles same-size images on a grid, row by row; missing cells stay black.

    Args:
        tiles: (H, W, 3) images.
        columns: Images per row.

    Returns:
        The sheet.
    """
    height, width = tiles[0].shape[:2]
    rows = -(-len(tiles) // columns)
    sheet = np.zeros((rows * height, columns * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        sheet[row * height:(row + 1) * height, column * width:(column + 1) * width] = tile
    return sheet


def video_sheet(video: Path, frames: int = 9, tile_width: int = 448) -> bytes:
    """Samples frames evenly over a video and tiles them, as a JPEG.

    Args:
        video: Video file.
        frames: Number of frames.
        tile_width: Width of each frame on the sheet, in pixels.

    Returns:
        The JPEG-encoded contact sheet.

    Raises:
        RuntimeError: If the video cannot be read.
    """
    # OpenCV is only installed on the cluster.
    # pylint: disable-next=import-outside-toplevel,import-error
    import cv2  # type: ignore[import-not-found]

    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for index in ((np.arange(frames) + 0.5) * count / frames).astype(int):
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if ok:
            height = round(frame.shape[0] * tile_width / frame.shape[1])
            tiles.append(cv2.resize(frame, (tile_width, height), interpolation=cv2.INTER_AREA))
    if not tiles:
        raise RuntimeError(f"cannot read frames from {video}")
    ok, jpeg = cv2.imencode(".jpg", contact_sheet(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(jpeg)


def ask_api(config: MetaConfig, sheet: bytes, limit: int = MAX_PROMPTS,
            client: Any = None) -> str:
    """Asks the API's vision model (Muse Spark) for the inventory.

    Args:
        config: API configuration; ``vision_model`` is used.
        sheet: JPEG contact sheet.
        limit: Most kinds of objects asked for.
        client: SDK client; created from ``config`` if None.

    Returns:
        The model's answer.
    """
    client = client if client is not None else config.client()
    url = "data:image/jpeg;base64," + base64.b64encode(sheet).decode()
    effort = {"reasoning_effort": config.reasoning_effort} if config.reasoning_effort else {}
    response = call_with_retry(lambda: client.chat.completions.create(
        model=config.vision_model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": INVENTORY_PROMPT.format(limit=limit)},
            {"type": "image_url", "image_url": {"url": url}}]}],
        **effort,
    ))
    return str(response.choices[0].message.content or "")


def ask_open(sheet: bytes, limit: int = MAX_PROMPTS, model: str = OPEN_VISION_MODEL) -> str:
    """Asks Llama 3.2 Vision, run with the released weights on the local GPU.

    Args:
        sheet: JPEG contact sheet.
        limit: Most kinds of objects asked for.
        model: Hugging Face model id; the weights must be in the local cache.

    Returns:
        The model's answer.
    """
    # Heavy GPU dependencies are only installed on the cluster (vLLM no longer
    # serves this model, transformers does).
    # pylint: disable=import-outside-toplevel,import-error
    import io

    import torch
    from PIL import Image
    from transformers import (  # type: ignore[import-not-found]
        AutoProcessor,
        MllamaForConditionalGeneration,
    )

    # pylint: enable=import-outside-toplevel,import-error
    processor = AutoProcessor.from_pretrained(model)
    llama = MllamaForConditionalGeneration.from_pretrained(model, torch_dtype=torch.bfloat16)
    llama.to("cuda")
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text",
                                              "text": INVENTORY_PROMPT.format(limit=limit)}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(images=Image.open(io.BytesIO(sheet)).convert("RGB"), text=prompt,
                       add_special_tokens=False, return_tensors="pt").to(llama.device)
    output = llama.generate(**inputs, max_new_tokens=200, do_sample=False)
    return str(processor.decode(output[0][inputs["input_ids"].shape[-1]:],
                                skip_special_tokens=True))


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.

    Raises:
        SystemExit: If the model named no object.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Prompt list to write.")
    parser.add_argument("--backend", choices=("api", "open"),
                        default=os.environ.get("PIVOT_BACKEND", "api").lower())
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=OPEN_VISION_MODEL, help="Model of the open backend.")
    parser.add_argument("--max-objects", type=int, default=MAX_PROMPTS,
                        help="Most kinds of objects listed (each costs SAM 3.1 a pass).")
    args = parser.parse_args(argv)

    sheet = video_sheet(args.video)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".jpg").write_bytes(sheet)
    if args.backend == "open":
        answer = ask_open(sheet, args.max_objects, args.model)
    else:
        answer = ask_api(MetaConfig.from_env(args.env_file), sheet, args.max_objects)
    prompts = parse_inventory(answer, args.max_objects)
    print(f"Model answer: {answer.strip()}", flush=True)
    if not prompts:
        raise SystemExit("The model named no object.")
    args.out.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    print(f"Objects to track: {', '.join(prompts)}", flush=True)


if __name__ == "__main__":
    main()
