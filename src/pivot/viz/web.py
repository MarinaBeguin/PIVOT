"""Standalone HTML viewer of a scene: point cloud, floor map and object boxes."""

import base64
import dataclasses
import json
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from pivot.geometry.types import FloatArray
from pivot.scene.graph import Scene
from pivot.scene.rooms import find_rooms
from pivot.viz.ownership import point_owners
from pivot.viz.pointcloud import ColorArray

_PLACEHOLDER = "__PIVOT_SCENE_DATA__"
# The texts of the page in every language (i18n.js), inlined like the data.
_LANGUAGES_PLACEHOLDER = "__PIVOT_LANGUAGES__"


def scene_payload(scene: Scene, points: FloatArray, colors: ColorArray,
                  title: str) -> dict[str, Any]:
    """Builds the JSON payload consumed by the viewer page.

    Args:
        scene: The scene graph.
        points: (N, 3) point cloud.
        colors: (N, 3) RGB colors.
        title: Page title.

    Returns:
        A JSON-compatible dictionary; arrays are base64-encoded binary.
    """
    payload: dict[str, Any] = {
        "title": title,
        "points": base64.b64encode(points.astype("<f4").tobytes()).decode(),
        "colors": base64.b64encode(colors.astype(np.uint8).tobytes()).decode(),
        "objects": [o.to_dict() for o in sorted(scene.objects.values(), key=lambda o: o.object_id)],
        # The movable object of each point (-1 for none): its points move with it.
        "owners": base64.b64encode(point_owners(scene, points).astype("<i4").tobytes()).decode(),
    }
    payload["rooms"] = []
    floor_data: dict[str, Any] | None = None
    if scene.floor is not None:
        floor = scene.floor
        floor_data = {
            "origin": floor.origin.tolist(),
            "resolution": floor.resolution,
            "rows": floor.shape[0],
            "cols": floor.shape[1],
            "cells": [np.flatnonzero(row).tolist() for row in floor.floor],
        }
        rooms = find_rooms(scene)
        if rooms is not None and len(rooms.rooms) > 1:
            # The room of each floor cell, in the order of "cells".
            floor_data["rooms"] = [rooms.labels[r, np.flatnonzero(row)].tolist()
                                   for r, row in enumerate(floor.floor)]
            payload["rooms"] = [{"id": room.room_id, "name": room.name,
                                 "center": room.center.tolist()} for room in rooms.rooms]
    payload["floor"] = floor_data
    return payload


class RoomMenu(NamedTuple):
    """The room menu of a page.

    Attributes:
        current: The id of the page's room.
        rooms: Every room the server offers, as ``(id, title)``.
    """

    current: str
    rooms: list[tuple[str, str]]


@dataclass(frozen=True)
class Features:
    """What the server behind a page offers; nothing for a page saved as a file.

    Attributes:
        interactive: The page is served by :mod:`pivot.server`, which plans
            and executes moves.
        chat: The server answers requests in words (agent).
        voice: Requests can also be spoken (microphone).
        render: The server draws realistic views of the room (Muse Image).
    """

    interactive: bool = False
    chat: bool = False
    voice: bool = False
    render: bool = False


def render_viewer(scene: Scene, points: FloatArray, colors: ColorArray,
                  title: str = "PIVOT scene", features: Features = Features(),
                  menu: RoomMenu | None = None) -> str:
    """Renders the self-contained viewer page of a scene.

    The page loads three.js from a CDN and embeds all scene data.

    Args:
        scene: The scene graph.
        points: (N, 3) point cloud.
        colors: (N, 3) RGB colors.
        title: Page title.
        features: What the server behind the page offers.
        menu: The rooms the server offers and this page's room, sent with
            every request; no menu if None.

    Returns:
        The HTML page.
    """
    template = resources.files("pivot.viz").joinpath("viewer.html").read_text(encoding="utf-8")
    payload = scene_payload(scene, points, colors, title)
    payload.update(dataclasses.asdict(features))
    payload["scenes"] = [{"id": i, "title": t} for i, t in menu.rooms] if menu else []
    payload["scene"] = menu.current if menu else ""
    # "</" must not appear inside the inline JSON script element.
    languages = resources.files("pivot.viz").joinpath("i18n.js").read_text(encoding="utf-8")
    return (template.replace(_LANGUAGES_PLACEHOLDER, languages)
            .replace(_PLACEHOLDER, json.dumps(payload).replace("</", "<\\/")))


def export_viewer(scene: Scene, points: FloatArray, colors: ColorArray, output: Path,
                  title: str = "PIVOT scene") -> Path:
    """Writes the static viewer page of a scene, see :func:`render_viewer`.

    Args:
        scene: The scene graph.
        points: (N, 3) point cloud.
        colors: (N, 3) RGB colors.
        output: Destination ``.html`` file.
        title: Page title.

    Returns:
        The output path.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_viewer(scene, points, colors, title), encoding="utf-8")
    return output
