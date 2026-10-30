"""Serves the PIVOT public demo on Modal: the viewer and the agent on a prebuilt scene.

    PIVOT_DEMO_SCENE=/path/to/out_dir/scene modal deploy deploy/modal_app.py   # from the repo root

The scene directory (scene.json and cloud.npz, from ``python -m pivot.build``)
is copied into the image: the demo never runs SAM, only the agent (Muse Spark)
and speech transcription, through the Meta API. At deploy time the API key is
read from the repository's .env and handed to Modal as a secret; nothing is
printed.

Spending stays bounded at three levels:
- the server prices every request and keeps a ledger on a Modal Volume, so a
  restart keeps counting: $25 in all, $1 and 20 requests per visitor per day
  (PIVOT_BUDGET_USD, PIVOT_VISITOR_DAILY_USD, PIVOT_VISITOR_DAILY_REQUESTS);
- one container at most, on one CPU, stopped after 15 idle minutes;
- spending limits set on the Modal workspace and on the Meta API account.
"""

from __future__ import annotations

import os
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
}


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
    return {"META_API_KEY": str(values["META_API_KEY"])}


def _scene_dir() -> Path:
    """Finds the scene to serve, at deploy time only.

    Returns:
        The directory holding scene.json and cloud.npz.

    Raises:
        SystemExit: If PIVOT_DEMO_SCENE is not set or holds no scene.
    """
    scene = Path(os.environ.get("PIVOT_DEMO_SCENE", ""))
    if not (scene / "scene.json").is_file() or not (scene / "cloud.npz").is_file():
        raise SystemExit("Set PIVOT_DEMO_SCENE to a scene directory (scene.json, cloud.npz).")
    return scene


app = modal.App("pivot-demo")
ledger = modal.Volume.from_name("pivot-ledger", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(ROOT / "pyproject.toml", "/root/pivot/pyproject.toml", copy=True)
    .add_local_file(ROOT / "README.md", "/root/pivot/README.md", copy=True)
    .add_local_dir(ROOT / "src", "/root/pivot/src", copy=True, ignore=["**/__pycache__/**"])
    .run_commands("pip install --no-cache-dir '/root/pivot[serve]'")
    .env(SETTINGS)
)
if modal.is_local():
    image = image.add_local_dir(_scene_dir(), "/scene")
secret = modal.Secret.from_dict(_secret() if modal.is_local() else {})


@app.function(image=image, secrets=[secret], volumes={"/ledger": ledger},
              min_containers=0, max_containers=1, scaledown_window=900,
              cpu=1.0, memory=2048, timeout=600)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web() -> FastAPI:
    """Creates the public application.

    Returns:
        The FastAPI application.
    """
    # pylint: disable-next=import-outside-toplevel
    from pivot.server.app import load_app

    return load_app(Path("/scene"), env_file=Path("/root/no.env"),
                    max_points=int(os.environ["PIVOT_MAX_POINTS"]), public=True)
