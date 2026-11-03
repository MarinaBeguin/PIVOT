"""Doors and windows from Meta's SceneScript, as an alternative to PIVOT's own.

SceneScript (Reality Labs) reads a room's layout from a point cloud and writes
it as a short program: ``make_wall``, ``make_door`` and ``make_window`` lines,
each with its parameters. It was trained on the semi-dense point clouds of
Project Aria glasses, which is where PIVOT should go next; on an iPhone scan
it may do better or worse than PIVOT's own doors and windows, so it is an
option to compare, not a replacement. Its code and weights are licensed
CC BY-NC (non-commercial) and need a CUDA GPU (see ``scripts/gpu``).

This module is the bridge, tested without the model: it writes a scene's
points in SceneScript's input format, and reads SceneScript's program back
into door and window objects that replace PIVOT's::

    python -m pivot.lifting.scenescript export <scene_dir> <points.csv.gz>
    python -m pivot.lifting.scenescript import <scene_dir> <layout.txt> <out_scene_dir>
"""

from __future__ import annotations

import argparse
import gzip
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.types import FloatArray
from pivot.scene.graph import Scene, SceneObject, is_opening

# A wall SceneScript gives no thickness to is taken this thick, in meters.
_MIN_THICKNESS = 0.1


def parse_program(text: str) -> list[dict[str, Any]]:
    """Reads a SceneScript program, one entity per line.

    Args:
        text: Lines like ``make_wall, id=0, a_x=1.0, ...``.

    Returns:
        One dictionary per entity: ``"command"`` and its numeric parameters.
    """
    entities = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if not parts or not parts[0].startswith("make_"):
            continue
        entity: dict[str, Any] = {"command": parts[0]}
        for part in parts[1:]:
            key, _, value = part.partition("=")
            try:
                entity[key.strip()] = float(value)
            except ValueError:
                continue
        entities.append(entity)
    return entities


def openings_from_program(text: str, first_id: int) -> list[SceneObject]:
    """Turns SceneScript's doors and windows into fixed objects set in their walls.

    Args:
        text: A SceneScript program.
        first_id: Id of the first object made.

    Returns:
        The doors and windows, as boxes as wide and tall as SceneScript says,
        as thick as their wall, turned along it.
    """
    entities = parse_program(text)
    walls = {int(e["id"]): e for e in entities if e["command"] == "make_wall" and "id" in e}
    objects: list[SceneObject] = []
    for entity in entities:
        label = {"make_door": "door", "make_window": "window"}.get(entity["command"])
        wall = walls.get(int(entity.get("wall0_id", -1)))
        if label is None or wall is None:
            continue
        yaw = math.atan2(wall["b_y"] - wall["a_y"], wall["b_x"] - wall["a_x"])
        thickness = max(float(wall.get("thickness", 0.0)), _MIN_THICKNESS)
        box = OrientedBox(np.array([entity["position_x"], entity["position_y"],
                                    entity["position_z"]]),
                          np.array([entity["width"], thickness, entity["height"]]), yaw)
        objects.append(SceneObject(first_id + len(objects), label, box, movable=False))
    return objects


def with_scenescript_openings(scene: Scene, text: str) -> Scene:
    """Replaces a scene's doors and windows with SceneScript's.

    Args:
        scene: The scene.
        text: SceneScript's program for the same room, in the scene's frame.

    Returns:
        The scene, its openings replaced (kept if SceneScript found none).
    """
    openings = openings_from_program(text, max(scene.objects, default=0) + 1)
    if not openings:
        return scene
    kept = [o for o in scene.objects.values() if not is_opening(o)]
    return Scene.from_objects([*kept, *openings], scene.floor,
                              {**scene.metadata, "openings": "SceneScript"}, scene.ceiling)


def write_points(points: FloatArray, path: Path) -> None:
    """Writes points in the format SceneScript reads (Project Aria semi-dense points).

    Args:
        points: (N, 3) points, z up.
        path: Destination ``.csv.gz`` file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as out:
        # Scans have no per-point uncertainty: every point passes SceneScript's filter.
        out.write("uid,graph_uid,px_world,py_world,pz_world,inv_dist_std,dist_std\n")
        for i, (x, y, z) in enumerate(points):
            out.write(f"{i},0,{x:.4f},{y:.4f},{z:.4f},0,0\n")


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: export points, or import SceneScript's layout.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description="Bridge to Meta's SceneScript.")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Write a scene's points for SceneScript.")
    export.add_argument("scene_dir", type=Path)
    export.add_argument("points", type=Path)
    load = commands.add_parser("import", help="Replace a scene's doors and windows.")
    load.add_argument("scene_dir", type=Path)
    load.add_argument("layout", type=Path)
    load.add_argument("out_dir", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        write_points(np.load(args.scene_dir / "cloud.npz")["points"], args.points)
        return
    scene = Scene.load(args.scene_dir / "scene.json")
    updated = with_scenescript_openings(scene, args.layout.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.scene_dir / "cloud.npz", args.out_dir / "cloud.npz")
    updated.save(args.out_dir / "scene.json")
    count = sum(is_opening(o) for o in updated.objects.values())
    print(f"{count} doors and windows from SceneScript in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
