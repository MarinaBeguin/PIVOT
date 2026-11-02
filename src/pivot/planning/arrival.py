"""A new piece of furniture arriving: it goes exactly where the user puts it.

Furniture bought new often comes flat-packed and is assembled where it will
stand: its first placement is not a move from the door, so no path is
checked, only that the spot itself is free (on the floor, clear of walls and
of other objects). If movable objects stand there, they can be moved aside,
each with a verified motion. Once there, the new object is like any other:
its later moves are planned and checked. Furniture measured in the scan never
arrives this way. Whether it would pass the doors assembled is told apart,
from the doors' widths: then it can also be delivered whole.

A new object never goes under another one (under a table, a desk): hidden
there, the user would not see it arrive. Its spot is checked against the
other objects' whole boxes, not the room left under their tops.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.planning.motion import MotionOptions, MotionPlan
from pivot.planning.placement import ConflictKind, check_placement
from pivot.planning.rearrange import clear_spot
from pivot.scene.graph import Scene, SceneObject

# Margin left on each side when an object is carried through a door, in meters.
DOOR_MARGIN = 0.02
_REASONS = {
    ConflictKind.OFF_FLOOR: "it is not on the floor there",
    ConflictKind.WALL_CLEARANCE: "it would go into a wall",
    ConflictKind.CEILING: "it is taller than the ceiling there",
}


@dataclass
class Arrival:
    """Result of :func:`arrive`.

    Attributes:
        placed: Whether the object stands at the spot.
        in_the_way: Ids of the objects standing in the spot.
        steps: Verified motions of the objects moved aside, in order.
        reason: Why it was not placed, if it was not.
    """

    placed: bool
    in_the_way: list[int] = field(default_factory=list)
    steps: list[MotionPlan] = field(default_factory=list)
    reason: str = ""


def with_new_object(scene: Scene, label: str, size: tuple[float, float, float], x: float,
                    y: float, yaw_deg: float = 0.0) -> tuple[Scene, int, OrientedBox]:
    """Adds a new object to a copy of a scene, standing on the floor at a spot.

    Args:
        scene: The scene.
        label: What it is.
        size: Width, depth and height, in meters.
        x: Where its center goes.
        y: Where its center goes.
        yaw_deg: Its orientation, in degrees.

    Returns:
        The scene with it, its id and its box.
    """
    object_id = max(scene.objects, default=0) + 1
    box = OrientedBox(np.array([x, y, 0.5 * size[2]]), np.array(size, dtype=float),
                      math.radians(yaw_deg))
    trial = Scene.from_objects([*scene.objects.values(), SceneObject(object_id, label, box)],
                               scene.floor, scene.metadata, scene.ceiling)
    return trial, object_id, box


def boxed(scene: Scene, object_id: int) -> Scene:
    """Fills the other objects' boxes, so that nothing fits under a table top.

    Args:
        scene: The scene, holding the new object.
        object_id: The new object.

    Returns:
        The scene, the other objects without their carved shapes.
    """
    return dataclasses.replace(scene, objects={
        i: o if i == object_id or o.shape is None else dataclasses.replace(o, shape=None)
        for i, o in scene.objects.items()})


def arrive(scene: Scene, object_id: int, box: OrientedBox, make_room: bool = False,
           options: MotionOptions | None = None) -> Arrival:
    """Puts a new object at an exact spot, without a path, moving others aside if asked.

    Args:
        scene: The scene, holding the new object (anywhere).
        object_id: The new object.
        box: The spot, as its box.
        make_room: Whether movable objects standing there are moved aside.
        options: Motion parameters of those moves.

    Returns:
        The arrival: placed, or what is in the way and why.
    """
    report = check_placement(boxed(scene, object_id), object_id, box)
    if report.feasible:
        return Arrival(True)
    in_the_way = sorted({c.other_id for c in report.conflicts if c.other_id is not None})
    refusal = _refusal(scene, report.conflicts, in_the_way, make_room)
    if refusal:
        return Arrival(False, in_the_way, reason=refusal)
    cleared = clear_spot(scene.without([object_id]), box, in_the_way, options)
    if not cleared.success:
        return Arrival(False, in_the_way, reason=cleared.message)
    after = scene
    for step in cleared.steps:
        after = after.moved(step.object_id, step.waypoints[-1])
    if not check_placement(boxed(after, object_id), object_id, box).feasible:
        return Arrival(False, in_the_way, reason="Even with the objects moved aside, the spot is "
                                                 "not free.")
    return Arrival(True, in_the_way, cleared.steps)


def _refusal(scene: Scene, conflicts: list[Any], in_the_way: list[int],
             make_room: bool) -> str:
    """Says why a spot cannot take a new object before anything is moved.

    Args:
        scene: The scene.
        conflicts: The spot's placement conflicts.
        in_the_way: Objects standing there.
        make_room: Whether movable objects may be moved aside.

    Returns:
        The reason, or an empty string if objects can be moved aside.
    """
    for conflict in conflicts:
        if conflict.kind in _REASONS:
            return f"The spot does not fit: {_REASONS[conflict.kind]}."
    fixed = [i for i in in_the_way if not scene[i].movable]
    if fixed:
        names = ", ".join(f"the {scene[i].label} (#{i})" for i in fixed)
        return f"{names} cannot be moved and stands there."
    if not make_room:
        names = ", ".join(f"the {scene[i].label} (#{i})" for i in in_the_way)
        verb = "stands" if len(in_the_way) == 1 else "stand"
        return f"{names} {verb} there: move it or make room."
    return ""


def doors_passed(scene: Scene, size: tuple[float, float, float]) -> list[dict[str, Any]]:
    """Says which doors the object would pass assembled, from their widths.

    A door's width is its long horizontal side; the object passes its
    narrowest horizontal side first, if it is not taller than the door. This
    compares sizes only: the way to the spot is not planned.

    Args:
        scene: The scene.
        size: Width, depth and height of the object, in meters.

    Returns:
        One entry per door: its id, width and whether the object fits through.
    """
    narrow = min(size[0], size[1])
    result = []
    for door in sorted(scene.objects.values(), key=lambda o: o.object_id):
        if door.label.lower() != "door":
            continue
        width = float(max(door.box.size[0], door.box.size[1]))
        result.append({"id": door.object_id, "width_m": round(width, 2),
                       "fits_assembled": narrow + 2 * DOOR_MARGIN <= width
                       and size[2] <= float(door.box.size[2])})
    return result
