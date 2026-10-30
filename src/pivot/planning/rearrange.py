"""Making room: moving blocking objects aside so that another one can pass.

When an object cannot reach its goal, other movable objects are often what
stands in the way (chairs in a passage). The obstacles that matter are found
by planning as if every other movable object were absent: those the object
would then run into are the blockers. Each blocker is parked at the nearest
spot outside the corridor the object will sweep, with a verified motion, and
the object's own motion is planned again in the cleared room. Then each
blocker is put back as close as possible to where it was (a chair pushed
aside goes back next to its desk). Every step is a regular, exactly verified
motion.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import box_distance_lower_bound, box_penetration
from pivot.planning.cspace import ConfigurationSpace
from pivot.planning.motion import (
    MotionOptions, MotionPlan, MotionStatus, densify, plan_motion, straight_motion,
)
from pivot.planning.placement import ConflictKind, check_placement
from pivot.planning.snap import nearest_feasible_pose, ranked_free_poses
from pivot.scene.graph import Scene, carried_objects


@dataclass
class Rearrangement:
    """Result of :func:`plan_rearrangement`.

    Attributes:
        success: Whether the object can reach its goal after the steps.
        steps: Verified motions, in execution order: blockers aside, the
            requested object, blockers back. Empty on failure.
        message: Human-readable explanation.
        blockers: Ids of the objects found in the way.
    """

    success: bool
    steps: list[MotionPlan] = field(default_factory=list)
    message: str = ""
    blockers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the result for tool-call responses.

        Returns:
            A JSON-compatible dictionary.
        """
        return {"success": self.success, "message": self.message, "blockers": self.blockers,
                "steps": [step.to_dict() for step in self.steps]}


def _without_movables(scene: Scene, object_id: int) -> Scene:
    """Removes every movable object except the moved one and what it carries.

    Args:
        scene: The scene.
        object_id: The object to move.

    Returns:
        The relaxed scene.
    """
    keep = {object_id, *carried_objects(scene, object_id)}
    objects = [o for o in scene.objects.values() if not o.movable or o.object_id in keep]
    return dataclasses.replace(scene, objects={o.object_id: o for o in objects})


def _blockers(scene: Scene, object_id: int, sweep: list[OrientedBox]) -> list[int]:
    """Lists the movable objects a motion runs into, in the order it meets them.

    Args:
        scene: The scene.
        object_id: The moved object.
        sweep: Densified poses of its motion.

    Returns:
        Ids of the blocking objects.
    """
    found: list[int] = []
    for box in sweep:
        for conflict in check_placement(scene, object_id, box).conflicts:
            other = conflict.other_id
            if (conflict.kind is ConflictKind.COLLISION and other is not None
                    and scene[other].movable and other not in found):
                found.append(other)
    return found


def _clear_of(box: OrientedBox, sweep: list[OrientedBox]) -> bool:
    """Tests whether a parked object stays out of another object's sweep.

    Boxes are compared, not volumes: a spot under the passing object (a chair
    under a table that is being moved) is not a parking spot.

    Args:
        box: The parked object's box.
        sweep: Densified poses of the moving object.

    Returns:
        True if no pose of the sweep overlaps the parked box.
    """
    return all(box_distance_lower_bound(box, pose) > 0.0 or box_penetration(box, pose) == 0.0
               for pose in sweep)


# Longest straight push of a blocker aside, in meters.
_MAX_PUSH = 2.0


def _park(scene: Scene, object_id: int, sweep: list[OrientedBox],
          options: MotionOptions, max_tries: int = 12) -> MotionPlan | None:
    """Moves a blocker to a nearby spot outside a sweep.

    First, as a person would, it is pushed straight aside, shortest push
    first; failing that, it goes to the nearest free spot by any path.

    Args:
        scene: The scene.
        object_id: The blocker.
        sweep: Densified poses of the object whose way must be cleared.
        options: Motion parameters.
        max_tries: Maximum number of free spots tried after the pushes.

    Returns:
        The blocker's motion, or None if it cannot be parked.
    """
    current = scene[object_id].box
    for distance in np.arange(0.2, _MAX_PUSH + 1e-9, 0.1):
        for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            box = current.moved(current.center[:2]
                                + distance * np.array([np.cos(angle), np.sin(angle)]))
            if _clear_of(box, sweep):
                plan = straight_motion(scene, object_id, box, options)
                if plan is not None:
                    return plan
    space = ConfigurationSpace.build(scene, object_id, clearance=options.clearance)
    tries = 0
    # Short moves first.
    for center, yaw in ranked_free_poses(space, current.center[:2], current.yaw):
        box = current.moved(center, yaw)
        if not _clear_of(box, sweep):
            continue
        plan = plan_motion(scene, object_id, box, options)
        if plan.status is MotionStatus.SUCCESS:
            return plan
        tries += 1
        if tries >= max_tries:
            break
    return None


def _put_back(scene: Scene, object_id: int, original: OrientedBox, options: MotionOptions,
              max_shift: float = 1.0) -> MotionPlan | None:
    """Moves an object back as close as possible to its original pose.

    Args:
        scene: The scene, with the object parked.
        object_id: The object.
        original: Its pose before it was moved aside.
        options: Motion parameters.
        max_shift: Largest distance from the original pose, in meters.

    Returns:
        The motion, or None if no nearby pose is reachable.
    """
    goal = nearest_feasible_pose(scene, object_id, original, max_shift, options.clearance)
    if goal is None:
        return None
    plan = plan_motion(scene, object_id, goal, options)
    return plan if plan.status is MotionStatus.SUCCESS else None


def plan_rearrangement(scene: Scene, object_id: int, goal: OrientedBox,
                       options: MotionOptions | None = None, max_blockers: int = 3,
                       put_back: bool = True) -> Rearrangement:
    """Plans the moves that bring an object to a goal, moving blockers aside if needed.

    Args:
        scene: The scene.
        object_id: The object to move.
        goal: Its target box.
        options: Motion parameters, shared by every step.
        max_blockers: Largest number of objects moved out of the way.
        put_back: Whether blockers return near their place afterwards.

    Returns:
        The verified steps, or the reason no rearrangement was found.
    """
    options = options or MotionOptions()
    direct = plan_motion(scene, object_id, goal, options)
    if direct.status is MotionStatus.SUCCESS:
        return Rearrangement(True, [direct], "The object can be moved directly.")
    relaxed = plan_motion(_without_movables(scene, object_id), object_id, goal, options)
    if relaxed.status is not MotionStatus.SUCCESS:
        return Rearrangement(False, message="Even with every movable object out of the way, "
                             f"the object cannot get there: {relaxed.message}")
    sweep = densify(relaxed.waypoints)
    blockers = _blockers(scene, object_id, sweep)
    if len(blockers) > max_blockers:
        return Rearrangement(False, message=f"{len(blockers)} objects are in the way; at most "
                             f"{max_blockers} are moved aside.", blockers=blockers)
    initial = scene
    steps = []
    for blocker in blockers:
        parking = _park(scene, blocker, sweep, options)
        if parking is None:
            return Rearrangement(False, message=f"The {scene[blocker].label} (#{blocker}) is in "
                                 "the way and no free spot nearby keeps the passage clear.",
                                 blockers=blockers)
        steps.append(parking)
        scene = scene.moved(blocker, parking.waypoints[-1])
    return _finish(initial, scene, object_id, goal, steps, blockers, options, put_back)


def _finish(initial: Scene, scene: Scene, object_id: int, goal: OrientedBox,
            steps: list[MotionPlan], blockers: list[int], options: MotionOptions,
            put_back: bool) -> Rearrangement:
    """Moves the object once the blockers are parked, then puts them back.

    Args:
        initial: The scene before anything moved.
        scene: The scene with the blockers parked.
        object_id: The object to move.
        goal: Its target box.
        steps: The parking motions so far.
        blockers: The parked objects, in order.
        options: Motion parameters.
        put_back: Whether blockers return near their place afterwards.

    Returns:
        The rearrangement.
    """
    final = plan_motion(scene, object_id, goal, options)
    if final.status is not MotionStatus.SUCCESS:
        return Rearrangement(False, message="After moving the objects in the way, the object "
                             f"still cannot get there: {final.message}", blockers=blockers)
    moved = ", ".join(f"the {scene[b].label} (#{b})" for b in blockers)
    message = f"Possible after moving {moved} out of the way."
    steps = [*steps, final]
    if put_back:
        returns = _put_all_back(scene.moved(object_id, final.waypoints[-1]), initial,
                                blockers, options)
        steps += returns
        if returns:
            names = ", ".join(f"the {scene[s.object_id].label} (#{s.object_id})" for s in returns)
            message += f" Then {names} put back as close as possible to its place."
    return Rearrangement(True, steps, message, blockers)


def _put_all_back(scene: Scene, initial: Scene, blockers: list[int],
                  options: MotionOptions) -> list[MotionPlan]:
    """Puts blockers back near their places, last moved first.

    Args:
        scene: The scene after the requested move.
        initial: The scene before anything moved.
        blockers: The parked objects, in the order they were moved aside.
        options: Motion parameters.

    Returns:
        The motions of the blockers that could be put back.
    """
    returns = []
    for blocker in reversed(blockers):
        back = _put_back(scene, blocker, initial[blocker].box, options)
        if back is not None:
            returns.append(back)
            scene = scene.moved(blocker, back.waypoints[-1])
    return returns
