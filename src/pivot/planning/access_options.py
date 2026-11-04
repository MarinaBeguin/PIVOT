"""Read-only, bounded comparisons of places for a person or wheelchair."""

from __future__ import annotations

from typing import Any

import numpy as np

from pivot.planning.handling import can_be_carried
from pivot.planning.motion import MotionOptions
from pivot.planning.way import WHEELCHAIR_SPOT, Way, clear_way, plan_way
from pivot.scene.graph import Scene

TABLES = {"table", "desk", "dining table", "coffee table"}


def suggest_places(scene: Scene, target_id: int | None = None, width: float = .9,
                   spot_size: tuple[float, float] = WHEELCHAIR_SPOT,
                   facing_object_id: int | None = None) -> dict[str, Any]:
    """Compares existing access, then at most four rearrangements, without edits.

    Results are ranked by handling effort, moved objects and route length.
    Candidate and motion budgets are explicit; absence is not impossibility.

    Args:
        scene: Current layout, kept unchanged.
        target_id: Optional preferred place to find alternatives near.
        width: Passage width in meters.
        spot_size: Depth and width of the place.
        facing_object_id: Optional facing reference.

    Returns:
        Ranked proposals and the scope of the bounded search, without execution.
    """
    label = scene[target_id].label.lower() if target_id is not None else "table"
    candidates = [obj for obj in scene.objects.values()
                  if obj.label.lower() in (TABLES if label in TABLES else {label})
                  and obj.object_id != target_id]
    origins = ([scene[target_id].box.center[:2]] if target_id is not None else
               [obj.box.center[:2] for obj in scene.objects.values() if obj.label == "door"])
    candidates.sort(key=lambda obj: (min((float(np.linalg.norm(obj.box.center[:2] - origin))
                                         for origin in origins), default=0), obj.object_id))
    results: list[tuple[tuple[float, ...], dict[str, Any]]] = []
    estimates: list[tuple[int, int]] = []

    def add(object_id: int, way: Way) -> None:
        effort = sum(1 if can_be_carried(scene[i]) else 3 for i in way.moved)
        distance = sum(float(np.linalg.norm(step.waypoints[-1].center[:2]
                                           - step.waypoints[0].center[:2])) for step in way.steps)
        data = {"id": object_id, "label": scene[object_id].label,
                "already_accessible": not way.steps, "moves_required": len(way.moved),
                "handling_score": effort, "furniture_distance_m": round(distance, 2),
                "requires_confirmation": bool(way.steps),
                "steps": [step.to_dict() for step in way.steps], **way.to_dict(scene, width)}
        results.append(((effort, len(way.moved), distance, data.get("path_length_m", 0)), data))

    for obj in candidates:
        way = clear_way(scene, obj.object_id, width, spot_size, facing_object_id)
        if way.success:
            add(obj.object_id, way)
        elif way.spot is not None:
            estimates.append((sum(1 if can_be_carried(scene[i]) else 3 for i in way.moved),
                              obj.object_id))
    attempted = 0
    if not results:
        for _, object_id in sorted(estimates, key=lambda item: item[0])[:4]:
            attempted += 1
            way = plan_way(scene, object_id, width, spot_size,
                           MotionOptions(max_expansions=20_000, max_repairs=3,
                                         refine_yaw_bins=0), facing_object_id)
            if way.success:
                add(object_id, way)
    results.sort(key=lambda item: item[0])
    return {"alternatives": [data for _, data in results[:3]],
            "existing_places_checked": len(candidates), "rearrangements_tried": attempted,
            "search_scope": "All matching existing places; at most four nearby rearrangements. "
                            "No global optimum or impossibility is established.",
            "executed": False}
