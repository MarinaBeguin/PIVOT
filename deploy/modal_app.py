"""Serves the PIVOT public demo on Modal: the viewer and the agent on prebuilt rooms.

    PIVOT_DEMO_SCENES="Office=/path/office/scene;Classroom=/path/classroom/scene" \
        modal deploy deploy/modal_app.py   # from the repo root

Each room is a title and a scene directory (scene.json and cloud.npz, from
``python -m pivot.build``), the first one shown by default; the directories
are copied into the image. With uploads enabled, separate GPU/CPU jobs reconstruct visitors' videos. The web service runs the agent
(Muse Spark), speech transcription and objects brought in from a photo (Muse
Spark on the photo, SAM 3.1 on one image), through the Meta API. At deploy time the API key is
read from the repository's .env and handed to Modal as a secret; nothing is
printed.

Spending stays bounded at three levels:
- the server prices every request and keeps a ledger on a Modal Volume, so a
  restart keeps counting: $25 in all, $1 and 20 requests per visitor per day
  (PIVOT_BUDGET_USD, PIVOT_VISITOR_DAILY_USD, PIVOT_VISITOR_DAILY_REQUESTS);
- one container at most, on one CPU, stopped after 15 idle minutes;
- spending limits set on the Modal workspace and on the Meta API account.

With ``PIVOT_TTS=mms`` at serve or deploy time, answers are read aloud by MMS-TTS
(the same voice on every phone): the image then gets torch (CPU) and
transformers, and the voices, downloaded once at build time; the container
gets 4 GB instead of 2.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import modal

if TYPE_CHECKING:
    from fastapi import FastAPI

ROOT = Path(__file__).resolve().parent.parent
SETTINGS = {
    "PIVOT_BACKEND": "api",
    "PIVOT_BUDGET_USD": "25",
    "PIVOT_VISITOR_DAILY_USD": "1",
    "PIVOT_VISITOR_DAILY_REQUESTS": "20",
    "PIVOT_LEDGER": "/ledger/spend.json",
    # Modal's web proxy appends the visitor's address to X-Forwarded-For.
    "PIVOT_PROXY_HOPS": "1",
    "PIVOT_MAX_POINTS": "200000",
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": "/root/pivot/src:/root/pivot/deploy",
    "NUMBA_CACHE_DIR": "/tmp/pivot-numba",
    "PIVOT_MODAL_UPLOADS": os.environ.get("PIVOT_MODAL_UPLOADS", "1"),
}

for setting in ("PIVOT_UPLOAD_FRAMES", "PIVOT_UPLOAD_OBJECT_KINDS", "PIVOT_UPLOAD_MAX_BYTES",
                "PIVOT_UPLOAD_MAX_SECONDS", "PIVOT_UPLOAD_DAILY", "PIVOT_UPLOAD_PER_VISITOR",
                "PIVOT_UPLOAD_ACTIVE", "PIVOT_UPLOAD_RETENTION", "PIVOT_UPLOAD_NAMESPACE"):
    if setting in os.environ:
        SETTINGS[setting] = os.environ[setting]


def _secret() -> dict[str, str]:
    """Reads the API key from the repository's .env, at deploy time only.

    Returns:
        The variables the server needs.

    Raises:
        SystemExit: If no key is found.
    """
    from dotenv import dotenv_values  # pylint: disable=import-outside-toplevel

    values = {**dotenv_values(ROOT / ".env"), **os.environ}
    if not values.get("META_API_KEY"):
        raise SystemExit("META_API_KEY is missing: set it in .env or in the environment.")
    secrets = {"META_API_KEY": str(values["META_API_KEY"])}
    # Opened once as ?owner=<key>, it lifts the daily limits of the owner's browser.
    if values.get("PIVOT_OWNER_KEY"):
        secrets["PIVOT_OWNER_KEY"] = str(values["PIVOT_OWNER_KEY"])
    return secrets


def _rooms() -> list[tuple[str, Path]]:
    """Reads the rooms to serve from PIVOT_DEMO_SCENES, at deploy time only.

    Returns:
        Each room's title and scene directory, the default first.

    Raises:
        SystemExit: If no room is given, or a directory holds no scene.
    """
    rooms = []
    for spec in filter(None, os.environ.get("PIVOT_DEMO_SCENES", "").split(";")):
        title, _, path = spec.rpartition("=")
        scene = Path(path)
        if not title or not (scene / "scene.json").is_file() or not (scene / "cloud.npz").is_file():
            raise SystemExit(f"Not a room (Title=scene directory): {spec!r}")
        rooms.append((title, scene))
    if not rooms:
        raise SystemExit('Set PIVOT_DEMO_SCENES, e.g. "Office=/path/to/scene".')
    return rooms


app = modal.App("pivot-demo")
ledger = modal.Volume.from_name("pivot-ledger", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(ROOT / "pyproject.toml", "/root/pivot/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", "/root/pivot/README.md", copy=True)
    .add_local_dir(ROOT / "src", "/root/pivot/src", copy=True, ignore=["**/__pycache__/**"])
    .add_local_dir(ROOT / "scripts", "/root/pivot/scripts", copy=True,
                   ignore=["site.env", "**/__pycache__/**"])
    .add_local_dir(ROOT / "docker", "/root/pivot/docker", copy=True)
    .add_local_dir(ROOT / "deploy", "/root/pivot/deploy", copy=True,
                   ignore=["**/__pycache__/**"])
    .apt_install("ffmpeg")
    .run_commands("pip install --no-cache-dir '/root/pivot[serve,fast]' "
                  "'opencv-python-headless==4.10.0.84'", "pip check",
                  'python -c "import cv2; from pivot.perception.inventory import video_sheet"')
    .env(SETTINGS)
)
# Read here when serving or deploying, and from the image's environment in the container.
VOICE = os.environ.get("PIVOT_TTS", "").lower() == "mms"
if VOICE:
    image = (image
             .run_commands("pip install --no-cache-dir torch --index-url "
                           "https://download.pytorch.org/whl/cpu",
                           "pip install --no-cache-dir 'transformers>=4.45'")
             .run_commands('python -c "from pivot.server.voice import MMS_VOICES; '
                           'from transformers import AutoTokenizer, VitsModel; '
                           '[(AutoTokenizer.from_pretrained(m), VitsModel.from_pretrained(m)) '
                           'for m in MMS_VOICES.values()]"')
             .env({"PIVOT_TTS": "mms"}))
if modal.is_local():
    ROOMS = _rooms()
    image = image.env({"PIVOT_ROOMS": ";".join(f"{title}=/scenes/{index}"
                                              for index, (title, _) in enumerate(ROOMS))})
    for index, (_, scene_dir) in enumerate(ROOMS):
        image = image.add_local_dir(scene_dir, f"/scenes/{index}")
secret = modal.Secret.from_dict(_secret() if modal.is_local() else {})


# Public video processing is independently quota-limited, not billed to the chat ledger.
UPLOADS = os.environ.get("PIVOT_MODAL_UPLOADS", "1") == "1"
upload_factory = None
web_volumes = {"/ledger": ledger}
if UPLOADS:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "deploy"))
    from modal_video import attach_uploads

    upload_factory, video_volume = attach_uploads(app, image, secret)
    web_volumes["/videos"] = video_volume


@app.function(image=image, secrets=[secret], volumes=web_volumes,
              min_containers=0, max_containers=1, scaledown_window=900,
              cpu=1.0, memory=4096 if VOICE or UPLOADS else 2048, timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web() -> FastAPI:
    """Creates the public application.

    Returns:
        The FastAPI application.
    """
    # pylint: disable-next=import-outside-toplevel
    from pivot.server.app import load_app

    return load_app(os.environ["PIVOT_ROOMS"].split(";"), env_file=Path("/root/no.env"),
                    max_points=int(os.environ["PIVOT_MAX_POINTS"]), public=True,
                    uploads=upload_factory() if upload_factory else None)
