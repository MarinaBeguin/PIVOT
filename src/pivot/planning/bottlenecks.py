"""Measured furniture gaps on a candidate route, without claiming global impossibility."""

from __future__ import annotations

from typing import Any
from functools import partial

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import polygon_distance
from pivot.geometry.types import FloatArray
from pivot.scene.graph import Scene, SceneObject, is_flat, is_opening


def _distance_to(footprint: FloatArray, obj: SceneObject) -> float:
    """Measures an object's distance to a footprint.

    Args:
        footprint: Probe footprint.
        obj: Candidate obstacle.

    Returns:
        Polygon separation in meters.
    """
    return polygon_distance(footprint, obj.box.footprint())


def route_gaps(scene: Scene, path: FloatArray, width: float,
               ignore: int) -> list[dict[str, Any]]:
    """Reports close, opposing furniture pairs beside a proposed floor route.

    The measured value is the gap between footprints, not a proof that every
    other route or orientation is impossible. The caller must keep this scope.

    Args:
        scene: Scene with bounding footprints.
        path: Candidate route points in world coordinates.
        width: Required passage width in meters.
        ignore: Object identifier omitted from the diagnostic.

    Returns:
        Measured candidate gaps, with explicit limits on their interpretation.
    """
    obstacles = [obj for obj in scene.objects.values() if obj.object_id != ignore
                 and not (is_opening(obj) or is_flat(obj)) and obj.box.z_min < 1.9]
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for point in path[::max(1, len(path) // 80)]:
        probe = OrientedBox(np.array([*point[:2], .5]), np.array([.001, .001, 1.]), 0.)
        nearest = sorted(obstacles, key=partial(_distance_to, probe.footprint()))[:3]
        for index, first in enumerate(nearest):
            for second in nearest[index + 1:]:
                if float((first.box.center[:2] - point[:2])
                         @ (second.box.center[:2] - point[:2])) >= 0:
                    continue
                gap = polygon_distance(first.box.footprint(), second.box.footprint())
                if gap >= width:
                    continue
                key = (min(first.object_id, second.object_id),
                       max(first.object_id, second.object_id))
                found[key] = {"objects": [{"id": obj.object_id, "label": obj.label}
                                           for obj in (first, second)],
                              "gap_m": round(float(gap), 3), "required_width_m": width,
                              "scope": "candidate route; other routes not ruled out"}
    return sorted(found.values(), key=lambda item: item["gap_m"])[:3]
