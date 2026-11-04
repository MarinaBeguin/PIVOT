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
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import box_distance, box_distance_lower_bound, box_penetration
from pivot.geometry.types import BoolArray, FloatArray
from pivot.planning.cspace import ConfigurationSpace
from pivot.planning.access import preserves_doors
from pivot.planning.handling import can_be_carried
from pivot.planning.lift import carry_height, plan_carried_motion
from pivot.planning.parking_motion import carry_between
from pivot.planning.parking_pockets import parking_pockets
from pivot.planning.motion import (
    WEDGED, MotionOptions, MotionPlan, MotionStatus, check_stop, densify, plan_motion,
    straight_motion,
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
# Distance from a blocker's start within which free states are its way out, in meters.
_WAY_OUT = 0.3
# Search budget of each parking motion.
_PARK_EXPANSIONS = 20_000
# Light objects this close to a pushed table move with it, in meters.
_GROUP_GAP = 0.15
# Objects whose bottom is lower than this stand on the floor, in meters.
_ON_FLOOR = 0.1
# Most objects pushed together.
_MAX_GROUP = 10


def _reachable(space: ConfigurationSpace, start: OrientedBox) -> BoolArray:
    """Finds the free configurations connected to an object's start, on the lattice.

    A start squeezed against another object (a chair under a table) is often
    not free itself: the free states near it are taken as its way out.

    Args:
        space: Free configurations of the object.
        start: Its current box.

    Returns:
        A (yaw_bins, rows, cols) mask; every free state if none is near the start.
    """
    labels, _ = ndimage.label(space.free, structure=np.ones((3, 3, 3), dtype=bool))
    row, col = space.nearest_cell(start.center[:2])
    reach = int(np.ceil(_WAY_OUT / space.floor.resolution))
    near = labels[:, max(0, row - reach):row + reach + 1, max(0, col - reach):col + reach + 1]
    found = np.unique(near[near > 0])
    if found.size == 0:
        return space.free
    result: BoolArray = np.isin(labels, found)
    return result


def _group(scene: Scene, object_id: int) -> list[int]:
    """Lists an object and the light objects standing against it (a table and its chairs).

    Args:
        scene: The scene.
        object_id: The object.

    Returns:
        Its id first, then theirs.
    """
    box = scene[object_id].box
    return [object_id, *(o.object_id for o in scene.objects.values()
                         if o.object_id != object_id and o.movable and can_be_carried(o)
                         and o.box.z_min < _ON_FLOOR and box_distance(box, o.box) <= _GROUP_GAP)]


def _pushed_along(scene: Scene, object_id: int, shift: FloatArray) -> list[int] | None:
    """Finds what moves when an object is pushed: its chairs, and the furniture it pushes.

    As when a row of tables is moved back, a table that would run into
    another movable object pushes it too, with that one's chairs; a wall or a
    fixed object stops the push.

    Args:
        scene: The scene.
        object_id: The object pushed.
        shift: (2,) translation, in meters.

    Returns:
        The objects moving together, the pushed one first; None if a wall or
        a fixed object stops them, or if too many would move.
    """
    members = _group(scene, object_id)
    checked: set[int] = set()
    while len(checked) < len(members):
        if len(members) > _MAX_GROUP:
            return None
        member = next(m for m in members if m not in checked)
        checked.add(member)
        rest = dataclasses.replace(scene, objects={
            i: o for i, o in scene.objects.items() if i == member or i not in members})
        box = scene[member].box.moved(scene[member].box.center[:2] + shift)
        for conflict in check_placement(rest, member, box, warn=False).conflicts:
            other = conflict.other_id
            if conflict.kind is not ConflictKind.COLLISION or other is None \
                    or not scene[other].movable:
                return None
            members += [m for m in _group(scene, other) if m not in members]
    return members


def _push_group(scene: Scene, object_id: int, sweep: list[OrientedBox],
                options: MotionOptions) -> list[MotionPlan] | None:
    """Pushes an object straight aside with its chairs and the furniture it pushes.

    A table is moved back with its chairs, and the tables behind it with
    theirs, all by the same shift, as people would rearrange a classroom.
    Each one's straight slide is checked against everything but the group,
    which moves rigidly; the one furthest ahead goes first, so none runs into
    another.

    Args:
        scene: The scene.
        object_id: The object to push.
        sweep: Poses of the object whose way must be cleared.
        options: Motion parameters.

    Returns:
        The motions, in order, or None if no push clears the sweep.
    """
    for distance in np.arange(0.2, _MAX_PUSH + 1e-9, 0.2):
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            shift = distance * np.array([np.cos(angle), np.sin(angle)])
            if not _clear_of(scene[object_id].box.moved(scene[object_id].box.center[:2]
                                                         + shift), sweep):
                continue
            members = _pushed_along(scene, object_id, shift)
            if members is None or len(members) == 1:
                continue
            verified = _verified_group(scene, members, shift, sweep, options)
            if verified is not None:
                return verified
    return None


def _verified_group(scene: Scene, members: list[int], shift: FloatArray,
                    sweep: list[OrientedBox], options: MotionOptions) -> list[MotionPlan] | None:
    """Checks a group translation in its actual sequential execution order.

    Args:
        scene: Complete scene, including the other group members.
        members: Furniture and chairs translated by the same amount.
        shift: Translation in meters.
        sweep: Corridor that must remain free.
        options: Motion validation settings.

    Returns:
        Fully verified sequential slides, or None if any intermediate move fails.
    """
    boxes = {member: scene[member].box.moved(scene[member].box.center[:2] + shift)
             for member in members}
    if not all(_clear_of(bounds, sweep) for bounds in boxes.values()):
        return None
    ordered = sorted(members, key=lambda member: -float(np.dot(
        scene[member].box.center[:2], shift)))
    current = scene
    verified = []
    for member in ordered:
        checked = straight_motion(current, member, boxes[member], options)
        if checked is None:
            attempted = plan_motion(current, member, boxes[member], dataclasses.replace(
                options, max_expansions=min(5000, options.max_expansions)))
            if attempted.status is MotionStatus.SUCCESS:
                checked = attempted
        if checked is None and can_be_carried(current[member]):
            carried = plan_carried_motion(current, member, boxes[member],
                                          carry_height(current, member), options)
            if carried.status is MotionStatus.SUCCESS:
                checked = carried
        if checked is None:
            return None
        verified.append(checked)
        current = current.moved(member, checked.waypoints[-1])
    return verified


@dataclass(frozen=True)
class _Lattice:
    """The free configurations of a blocker, and those connected to where it stands."""

    space: ConfigurationSpace
    reachable: BoolArray

    def worth_trying(self, box: OrientedBox) -> bool:
        """Tests whether the lattice shows a pose free and connected to the blocker.

        Args:
            box: The pose.

        Returns:
            True if exact checks of it can succeed.
        """
        row, col = self.space.nearest_cell(box.center[:2])
        return bool(self.reachable[self.space.nearest_bin(box.yaw), row, col])


def _push_straight(scene: Scene, object_id: int, sweep: list[OrientedBox],
                   options: MotionOptions, lattice: _Lattice) -> MotionPlan | None:
    """Pushes a blocker straight aside, shortest push first.

    Args:
        scene: The scene.
        object_id: The blocker.
        sweep: Poses of the object whose way must be cleared.
        options: Motion parameters.
        lattice: Its free configurations.

    Returns:
        The push, or None.
    """
    current = scene[object_id].box
    # A blocker squeezed against another object (a chair under a table) cannot slide
    # straight out: its start already overlaps, which only a planned motion allows.
    if not check_placement(scene, object_id, current, options.clearance, warn=False).feasible:
        return None
    for distance in np.arange(0.2, _MAX_PUSH + 1e-9, 0.1):
        for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            box = current.moved(current.center[:2]
                                + distance * np.array([np.cos(angle), np.sin(angle)]))
            if lattice.worth_trying(box) and _clear_of(box, sweep):
                plan = straight_motion(scene, object_id, box, options)
                if plan is not None:
                    return plan
    return None


def _move_to_free_spot(scene: Scene, object_id: int, sweep: list[OrientedBox],
                       options: MotionOptions, lattice: _Lattice,
                       max_tries: int, prefer_edges: bool = False,
                       accept: Callable[[Scene], bool] | None = None) -> MotionPlan | None:
    """Moves a blocker to the nearest free spot by any path, carried if it must.

    Args:
        scene: The scene.
        object_id: The blocker.
        sweep: Poses of the object whose way must be cleared.
        options: Motion parameters.
        lattice: Its free configurations.
        max_tries: Maximum number of spots tried.
        prefer_edges: Require peripheral parking for light obstacles cleared for people.
        accept: Optional final-layout circulation predicate.

    Returns:
        The motion, or None.
    """
    # A spot the lattice connects to the blocker is found fast, or not at all.
    options = dataclasses.replace(options, max_expansions=min(options.max_expansions,
                                                             _PARK_EXPANSIONS),
                                 refine_yaw_bins=options.yaw_bins)
    # What cannot slide out (a chair boxed in between tables) can be lifted out, if light.
    lift = max(carry_height(scene, object_id), scene[object_id].box.z_min) \
        if can_be_carried(scene[object_id]) else None
    wedged = False
    # Short moves first.
    for center, yaw in _parking_poses(scene, object_id, lattice, prefer_edges, accept, sweep):
        box = scene[object_id].box.moved(center, yaw)
        if not _clear_of(box, sweep) or (lift is None and not lattice.worth_trying(box)):
            continue
        if not wedged and lattice.worth_trying(box):
            plan = plan_motion(scene, object_id, box, options)
            if plan.status is MotionStatus.SUCCESS:
                return plan
            # No room to slide from where it stands: no other spot will do better.
            wedged = plan.message == f"The start {WEDGED}"
        if lift is not None:
            plan = plan_carried_motion(scene, object_id, box, lift, options)
            if plan.status is MotionStatus.SUCCESS:
                return plan
            if plan.status is MotionStatus.INVALID_START:
                departure = _pull_out_and_carry(scene, object_id, box, lift, options, lattice)
                if departure is not None:
                    return departure
            if prefer_edges and plan.status in (MotionStatus.INVALID_START,
                                                MotionStatus.INVALID_GOAL):
                departure = carry_between(scene,object_id,box,lift,options,lattice.space)
                if departure is not None:
                    return departure
        elif wedged:
            return None
        max_tries -= 1
        if max_tries <= 0:
            break
    return None


def _parking_poses(scene: Scene, object_id: int, lattice: _Lattice, edges: bool,
                   accept: Callable[[Scene], bool] | None = None,
                   sweep: list[OrientedBox] | None = None
                   ) -> Iterator[tuple[FloatArray, float]]:
    """Filters peripheral destinations before expensive motion verification."""
    current = scene[object_id].box
    boundary = parking_pockets(scene,object_id) if edges else None
    rejected: list[FloatArray] = []
    for center,yaw in ranked_free_poses(lattice.space,current.center[:2],current.yaw):
        check_stop()
        if boundary is not None and not boundary.accepts(current.moved(center,yaw)):
            continue
        if accept is not None:
            if sweep is not None and not _clear_of(current.moved(center,yaw),sweep):
                continue
            if any(np.linalg.norm(center-other)<.3 for other in rejected):
                continue
            if not accept(scene.moved(object_id,current.moved(center,yaw))):
                rejected.append(center)
                continue
        yield center,yaw


def _pull_out_and_carry(scene: Scene, object_id: int, goal: OrientedBox, height: float,
                        options: MotionOptions, lattice: _Lattice) -> MotionPlan | None:
    """Tries up to four nearby departures before lifting an object from under furniture.

    Args:
        scene: Complete scene before the move.
        object_id: Light object to extract.
        goal: Free parking pose.
        height: Bottom height during carrying.
        options: Bounded motion settings.
        lattice: Free on-floor poses connected to the initial pose.

    Returns:
        Joined verified slide, lift, carry and lowering, or no verified solution.
    """
    start = scene[object_id].box
    tried = 0
    for center, yaw in ranked_free_poses(lattice.space, start.center[:2], start.yaw):
        if np.linalg.norm(center - start.center[:2]) > 1.:
            break
        departure = start.moved(center, yaw)
        if not lattice.worth_trying(departure):
            continue
        lifted = plan_carried_motion(scene.moved(object_id, departure), object_id,
                                     goal, height, dataclasses.replace(
                                         options, max_expansions=min(5000, options.max_expansions)))
        if lifted.status is MotionStatus.INVALID_START:
            continue
        tried += 1
        if lifted.status is MotionStatus.SUCCESS:
            slide = plan_motion(scene, object_id, departure, dataclasses.replace(
                options, max_expansions=min(5000, options.max_expansions)))
            if slide.status is MotionStatus.SUCCESS:
                return dataclasses.replace(lifted,
                    waypoints=slide.waypoints + lifted.waypoints[1:],
                    expansions=slide.expansions + lifted.expansions,
                    min_clearance=(min(slide.min_clearance, lifted.min_clearance)
                                   if slide.min_clearance is not None
                                   and lifted.min_clearance is not None else None),
                    message="Verified departure, lift, carry and lowering.")
        if tried == 4:
            break
    return None


def park_aside(scene: Scene, object_id: int, sweep: list[OrientedBox],
               options: MotionOptions, max_tries: int = 3,
               prefer_edges: bool = False,
               accept: Callable[[Scene], bool] | None = None) -> list[MotionPlan] | None:
    """Moves a blocker to a nearby spot outside a sweep.

    First, as a person would, it is pushed straight aside, shortest push
    first; then, if it cannot be carried (a table), pushed together with the
    chairs against it; failing that, it goes to the nearest free spot by any
    path. Only spots the lattice shows free and connected to the blocker are
    tried: the exact checks are kept for the ones that can succeed.

    Args:
        scene: The scene.
        object_id: The blocker.
        sweep: Densified poses of the object whose way must be cleared.
        options: Motion parameters.
        max_tries: Maximum number of free spots tried after the pushes.
        prefer_edges: Park light obstacles near boundaries rather than in open circulation.
        accept: Optional predicate preserving circulation in the candidate layout.

    Returns:
        The motions (the blocker's, and those of the chairs pushed with it),
        or None if it cannot be parked.
    """
    # Prefer a coherent group. A single small adjustment is still allowed if
    # it preserves the nearby seats' relation to their table.
    if not can_be_carried(scene[object_id]) and len(_group(scene, object_id)) > 1:
        grouped = _push_group(scene, object_id, sweep, options)
        if grouped is not None and _accepted(scene, grouped, accept):
            return grouped
    space = ConfigurationSpace.build(scene, object_id, clearance=options.clearance)
    lattice = _Lattice(space, _reachable(space, scene[object_id].box))
    pushed = (None if prefer_edges and can_be_carried(scene[object_id])
              else _push_straight(scene, object_id, sweep, options, lattice))
    if pushed is not None and _keeps_group(scene, object_id, pushed) \
            and _accepted(scene, [pushed], accept):
        return [pushed]
    if not can_be_carried(scene[object_id]) and len(_group(scene, object_id)) == 1:
        group = _push_group(scene, object_id, sweep, options)
        if group is not None and _accepted(scene, group, accept):
            return group
    moved = _move_to_free_spot(scene, object_id, sweep, options, lattice, max_tries,
                               prefer_edges and can_be_carried(scene[object_id]), accept)
    return None if moved is None or not _keeps_group(scene, object_id, moved) else [moved]


def _accepted(scene: Scene, plans: list[MotionPlan], accept: Callable[[Scene], bool] | None
               ) -> bool:
    """Checks a complete candidate group layout before choosing its parking plan."""
    if accept is None:
        return True
    for plan in plans:
        scene = scene.moved(plan.object_id,plan.waypoints[-1])
    return accept(scene)


def _keeps_group(scene: Scene, object_id: int, plan: MotionPlan) -> bool:
    """Allows small desk adjustments without abandoning its adjacent light furniture."""
    if can_be_carried(scene[object_id]):
        return True
    return all(box_distance(plan.waypoints[-1], scene[member].box) <= 2 * _GROUP_GAP
               for member in _group(scene, object_id)[1:])


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


def restore_group(scene: Scene, initial: Scene, object_id: int, sweep: list[OrientedBox],
                  options: MotionOptions, accept: Callable[[Scene],bool] | None = None
                  ) -> list[MotionPlan] | None:
    """Tries a partial return of a displaced furniture group without closing its passage.

    Args:
        scene: Current proposed layout.
        initial: Layout before rearrangement.
        object_id: Heavy furniture whose adjacent chairs moved with it.
        sweep: Reserved route and destination that must stay clear.
        options: Motion search settings.
        accept: Optional final-layout circulation predicate.

    Returns:
        Verified sequential return motions, or no safe improvement. Independently
        parked chairs are never dragged back to their original aisle.
    """
    if can_be_carried(initial[object_id]):
        return None
    members = _group(initial,object_id)
    shift = initial[object_id].box.center[:2]-scene[object_id].box.center[:2]
    if np.linalg.norm(shift)<.2 or any(
            not np.allclose(initial[i].box.center[:2]-scene[i].box.center[:2],shift,atol=.03)
            or abs(np.sin(initial[i].box.yaw-scene[i].box.yaw))>.03 for i in members):
        return None
    for fraction in (1.,.75,.5,.25):
        returned = _verified_group(scene,members,shift*fraction,sweep,options)
        if returned is not None and _accepted(scene,returned,accept):
            return returned
    return None


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
    steps: list[MotionPlan] = []
    for blocker in blockers:
        parking = park_aside(scene, blocker, sweep, options)
        if parking is None:
            return Rearrangement(False, message=f"The {scene[blocker].label} (#{blocker}) is in "
                                 "the way and no free spot nearby keeps the passage clear.",
                                 blockers=blockers)
        for motion in parking:
            steps.append(motion)
            scene = scene.moved(motion.object_id, motion.waypoints[-1])
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
    final_scene = initial
    for step in steps:
        final_scene = final_scene.moved(step.object_id, step.waypoints[-1])
    if not preserves_doors(initial, final_scene):
        return Rearrangement(False, message="The proposed rearrangement would obstruct a "
                             "door; no moves executed.", blockers=blockers)
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


def clear_spot(scene: Scene, spot: OrientedBox, blockers: list[int],
               options: MotionOptions | None = None) -> Rearrangement:
    """Moves objects aside so that a spot becomes free, each with a verified motion.

    Used when a new piece of furniture arrives where others stand: they are
    pushed aside, nearest free spot first, out of the spot; they are not put
    back (the spot is now taken).

    Args:
        scene: The scene, without the arriving object.
        spot: The box the arriving object will take.
        blockers: Objects standing in the spot.
        options: Motion parameters.

    Returns:
        The verified steps, or the reason the spot cannot be cleared.
    """
    options = options or MotionOptions()
    carried = {c for b in blockers for c in carried_objects(scene, b)}
    steps: list[MotionPlan] = []
    for blocker in (b for b in blockers if b not in carried):
        if any(step.object_id == blocker for step in steps):
            continue  # Already pushed aside with a table.
        parking = park_aside(scene, blocker, [spot], options)
        if parking is None:
            return Rearrangement(False, message=f"The {scene[blocker].label} (#{blocker}) stands "
                                 "there and no free spot nearby can take it.", blockers=blockers)
        for motion in parking:
            steps.append(motion)
            scene = scene.moved(motion.object_id, motion.waypoints[-1])
    moved = ", ".join(f"the {scene[s.object_id].label} (#{s.object_id})" for s in steps)
    return Rearrangement(True, steps, f"Room made by moving {moved} aside.", blockers)
