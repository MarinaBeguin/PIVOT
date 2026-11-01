"""Photo-like views of a rearranged room, drawn by Muse Image.

The viewer shows the room as colored points and blocks: true to the geometry,
hard to picture. Before freezing a plan, the visitor can ask for a realistic
view: the current 3D view (the moved objects carry their own points along) is
sent to Muse Image, which edits it into a photograph of the same room from the
same viewpoint, keeping every object where the plan puts it.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx
from openai.types import ImagesResponse

from pivot.meta.client import MetaConfig, call_with_retry

RENDER_PROMPT = (
    "@Image1 is a rough 3D view of a real room, rebuilt from a phone video: colored points, "
    "dark gaps where nothing was seen, and the furniture where the user has just moved it. "
    "Turn it into a realistic photograph of the same room, from exactly the same viewpoint. "
    "Keep every piece of furniture exactly where it is, with its size, shape and colors, and "
    "keep the walls, windows, doors and floor as they are. Fill the gaps plausibly and add "
    "nothing else: no new furniture, no people, no text.")


def render_prompt(references: list[str]) -> str:
    """Words the edit: the view, then a photo of each object brought in.

    Args:
        references: What each extra image shows, e.g. ``"the new armchair (#201)"``.

    Returns:
        The prompt; extra images are ``@Image2``, ``@Image3``...
    """
    extra = "".join(f" @Image{k} shows {what}: draw it as it looks in that photo."
                    for k, what in enumerate(references, start=2))
    return RENDER_PROMPT + extra


def photo_view(config: MetaConfig, view_png: bytes,
               references: list[tuple[str, bytes]] | None = None, client: Any = None,
               get: Any = httpx.get) -> bytes:
    """Edits a 3D view of the room into a photograph with Muse Image.

    Args:
        config: API configuration; ``image_model`` is used.
        view_png: The 3D view, as a PNG.
        references: Photos of objects brought into the room, as ``(what it
            shows, JPEG or PNG bytes)``.
        client: SDK client; created from ``config`` if None.
        get: HTTP GET function, for answers given as an address; injectable for tests.

    Returns:
        The image, as encoded bytes (PNG or JPEG).

    Raises:
        RuntimeError: If the answer holds no image.
    """
    client = client if client is not None else config.client()
    references = references or []
    images = [("view.png", view_png, "image/png"),
              *((f"reference{k}.png", data, _media_type(data))
                for k, (_, data) in enumerate(references, start=2))]
    # The API wants the images as image[0], image[1]...; the SDK's images.edit
    # sends a list as image[], which it refuses.
    response = call_with_retry(lambda: client.post(
        "/images/edits", cast_to=ImagesResponse,
        body={"model": config.image_model,
              "prompt": render_prompt([what for what, _ in references])},
        files=[(f"image[{k}]", image) for k, image in enumerate(images)],
        options={"headers": {"Content-Type": "multipart/form-data"}}))
    item = response.data[0] if response.data else None
    if item is not None and getattr(item, "b64_json", None):
        return base64.b64decode(item.b64_json)
    if item is not None and getattr(item, "url", None):
        answer = get(item.url, timeout=60.0)
        answer.raise_for_status()
        return bytes(answer.content)
    raise RuntimeError("Muse Image returned no image.")


def _media_type(data: bytes) -> str:
    """Tells a PNG from a JPEG by its first bytes.

    Args:
        data: Encoded image.

    Returns:
        The media type.
    """
    return "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
