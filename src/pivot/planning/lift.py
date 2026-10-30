"""Gravity and carried moves: objects rest on surfaces and can be lifted.

An object put down somewhere rests on the first surface below it that can
hold it (a desk top, a deep enough shelf; not a trash can or a chair), or on
the floor. When its height
changes, or when it is carried over furniture, the move has three parts: lift
it straight up, carry it at that height (everything lower is passed over), and
lower it straight down. Each part is verified; the person carrying the object
is not modeled.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import footprint_overlap
from pivot.planning.motion import MotionOptions, MotionPlan, MotionStatus, plan_motion
from pivot.planning.placement import check_placement
from pivot.scene.graph import SURFACE_LABELS, Scene, carried_boxes, carried_objects, is_opening


def resting_box(scene: Scene, object_id: int, box: OrientedBox, below: float,
                min_support: float = 0.6) -> OrientedBox:
    """Drops an object onto the first surface below a height that can hold it.

    Args:
        scene: The scene.
        object_id: The object.
        box: Its pose; only its footprint matters.
        below: Height the object falls from (its bottom), in meters.
        min_support: Fraction of the footprint that must lie over a surface.

    Returns:
        The box resting on the highest such surface, or on the floor.
    """
    movers = {object_id, *carried_objects(scene, object_id)}
    tops = [o.box.z_max for o in scene.objects.values()
            if o.object_id not in movers and o.label.lower() in SURFACE_LABELS
            and o.box.z_max <= below + 1e-6 and footprint_overlap(box, o.box) >= min_support]
    return box.moved(z_min=max(tops, default=0.0))


# Furniture up to this height is carried over (desks, low cabinets), not higher.
MAX_CARRIED_OVER = 1.1
# Room left under a carried object.
CARRY_MARGIN = 0.05


def carry_height(scene: Scene, object_id: int) -> float:
    """Chooses the height to carry an object at: just over the low furniture.

    Args:
        scene: The scene.
        object_id: The object.

    Returns:
        The height of the object's bottom while carried, in meters.
    """
    movers = {object_id, *carried_objects(scene, object_id)}
    tops = [o.box.z_max for o in scene.objects.values()
            if o.object_id not in movers and not is_opening(o)
            and o.box.z_max <= MAX_CARRIED_OVER]
    return max(tops, default=0.0) + CARRY_MARGIN


def _column(box: OrientedBox, z_from: float, z_to: float) -> OrientedBox:
    """Returns the box swept by moving a box vertically between two bottom heights.

    Args:
        box: The box.
        z_from: Bottom height at one end, in meters.
        z_to: Bottom height at the other end, in meters.

    Returns:
        The swept box.
    """
    low, high = min(z_from, z_to), max(z_from, z_to) + box.size[2]
    center = box.center.copy()
    center[2] = 0.5 * (low + high)
    return OrientedBox(center, np.array([*box.size[:2], high - low]), box.yaw)


def _vertical_ok(scene: Scene, object_id: int, box: OrientedBox, z_from: float,
                 z_to: float) -> bool:
    """Tests whether an object can move straight up or down without hitting anything.

    The swept column is tested as a solid box, ignoring the object's shape:
    conservative, and independent of where the shape's cells fall.

    Args:
        scene: The scene, with the object anywhere (its current pose is ignored).
        object_id: The object.
        box: Its pose (the height is ignored).
        z_from: Bottom height at the start, in meters.
        z_to: Bottom height at the end, in meters.

    Returns:
        True if the column is free.
    """
    column = _column(box, z_from, z_to)
    solid = dataclasses.replace(scene, objects={**scene.objects, object_id: dataclasses.replace(
        scene[object_id], box=column, shape=None)})
    # The surfaces the object leaves and reaches touch the column's ends.
    return check_placement(solid, object_id, column, collision_tolerance=0.02).feasible


def plan_carried_motion(scene: Scene, object_id: int, goal: OrientedBox, carry_bottom: float,
                        options: MotionOptions | None = None) -> MotionPlan:
    """Plans a lift, a carry at a given height, and a lowering onto the goal.

    Args:
        scene: The scene.
        object_id: The object.
        goal: Its final pose, resting on its support.
        carry_bottom: Height of the object's bottom while carried, in meters;
            at least the start and goal heights.
        options: Motion parameters of the carried part.

    Returns:
        The plan; its waypoints go up, across, and down.
    """
    start = scene[object_id].box
    carry_bottom = max(carry_bottom, start.z_min, goal.z_min)
    lifted, lowered = start.moved(z_min=carry_bottom), goal.moved(z_min=carry_bottom)
    tops = [lifted.z_max, *(b.z_max for b in carried_boxes(scene, object_id, lifted).values())]
    if scene.ceiling is not None and max(tops) > scene.ceiling:
        return MotionPlan(MotionStatus.INVALID_GOAL, object_id,
                          message=f"The object cannot be carried at that height: it would rise "
                                  f"{100 * (max(tops) - scene.ceiling):.0f} cm above the ceiling.")
    if not _vertical_ok(scene, object_id, start, start.z_min, carry_bottom):
        return MotionPlan(MotionStatus.INVALID_START, object_id,
                          message="The object cannot be lifted: something is above it.")
    if not _vertical_ok(scene, object_id, goal, goal.z_min, carry_bottom):
        return MotionPlan(MotionStatus.INVALID_GOAL, object_id,
                          message="The object cannot be lowered onto the goal: "
                                  "something is in the way above it.")
    carried = plan_motion(scene.moved(object_id, lifted), object_id, lowered, options)
    if carried.status is not MotionStatus.SUCCESS:
        return dataclasses.replace(carried, message=f"While carried: {carried.message}")
    waypoints = [start, *carried.waypoints, goal]
    return dataclasses.replace(carried, waypoints=waypoints,
                               message="A collision-free lift, carry and lowering was found.")
