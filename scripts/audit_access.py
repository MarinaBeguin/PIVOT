"""Write a read-only accessibility audit of any saved scene, without model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pivot.planning.access import check_access
from pivot.planning.way import way_is_clear
from pivot.scene.graph import Scene
from pivot.scene.rooms import find_rooms


def main() -> None:
    """Prints room topology, passage reports and already accessible tables as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--wall-evidence-from-cloud", action="store_true",
                        help="Derive the wall layer from cloud.npz in memory; change no scene file.")
    args = parser.parse_args()
    path = args.scene / "scene.json" if args.scene.is_dir() else args.scene
    scene = Scene.load(path)
    cloud_path = path.parent / "cloud.npz"
    if cloud_path.exists() or args.wall_evidence_from_cloud:
        with np.load(cloud_path) as cloud:
            scene = scene.with_wall_evidence(cloud["points"])
    rooms = find_rooms(scene)
    result = {
        "scene": str(path),
        "rooms": [room.name for room in rooms.rooms] if rooms is not None else [],
        "has_wall_evidence": scene.floor is not None and scene.floor.walls is not None,
        "door_leaves": [obj.object_id for obj in scene.objects.values()
                        if obj.label == "door leaf"],
        "access": {},
        "tables": [],
    }
    for width in (0.6, 0.9):
        report = check_access(scene, width)
        result["access"][str(width)] = report.to_dict() if report is not None else None
    for obj in scene.objects.values():
        if obj.label.lower() in {"table", "desk", "dining table", "coffee table"}:
            result["tables"].append({"id": obj.object_id, "label": obj.label,
                                     "wheelchair_route_and_spot": way_is_clear(scene, obj.object_id)})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
