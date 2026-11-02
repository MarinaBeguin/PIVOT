"""Moving a rug rolled up: what a person does when it does not pass flat.

A rug as wide as a room cannot go through a doorway flat, but rolled up it
is a roll as long as its short side and a few tens of centimeters thick.
The roll's path is planned like any motion (a rug is stopped only by walls);
the rug is unrolled at the goal, where it must lie flat.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.planning.motion import MotionOptions, MotionPlan, MotionStatus, plan_motion
from pivot.scene.graph import Scene

ROLL_DIAMETER = 0.3
"""Thickness of a rolled-up rug, in meters."""


def rolled_box(box: OrientedBox) -> OrientedBox:
    """Returns the box of a rug rolled up from one short edge to the other.

    Args:
        box: The rug lying flat.

    Returns:
        The roll: as long as the rug's short side, :data:`ROLL_DIAMETER` thick,
        at the same place and orientation, on the floor.
    """
    size = box.size.copy()
    long_side = int(np.argmax(size[:2]))
    size[long_side] = min(float(size[long_side]), ROLL_DIAMETER)
    size[2] = ROLL_DIAMETER
    center = box.center.copy()
    center[2] = box.z_min + 0.5 * ROLL_DIAMETER
    return OrientedBox(center, size, box.yaw)


def plan_rolled_motion(scene: Scene, object_id: int, goal: OrientedBox,
                       options: MotionOptions | None = None) -> MotionPlan:
    """Plans the move of a rug rolled up, unrolled at the goal.

    Call it when the rug does not pass flat: its goal, flat, is checked by
    that first plan.

    Args:
        scene: The scene; must have a floor map.
        object_id: The rug.
        goal: Where it lies at the end, flat.
        options: Query parameters.

    Returns:
        The roll's plan; on success, the rug flat at the goal is its last
        waypoint and ``rolled_up`` is set.
    """
    obj = scene[object_id]
    roll = dataclasses.replace(obj, box=rolled_box(obj.box), shape=None)
    rolled_scene = dataclasses.replace(scene, objects={**scene.objects, object_id: roll})
    plan = plan_motion(rolled_scene, object_id, rolled_box(goal), options)
    if plan.status is not MotionStatus.SUCCESS:
        return plan
    return dataclasses.replace(plan, waypoints=[*plan.waypoints, goal], rolled_up=True,
                               message=f"Rolled up to pass, then unrolled. {plan.message}")
