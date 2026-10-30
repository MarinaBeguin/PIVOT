"""Geometric tools exposed to the language model.

Every tool reads or edits a :class:`Session`, which holds the current scene
and its edit history. The model never edits the scene directly: an object only
moves through ``move_object``, which first plans and validates the motion.
All answers are JSON with numeric fields in meters and degrees, so the model
can reason about magnitudes rather than paraphrase verdicts.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import box_distance, polygon_distance
from pivot.planning.access import WALKING_WIDTH, check_access
from pivot.planning.handling import can_be_carried, people_needed
from pivot.planning.lift import carry_height, plan_carried_motion, resting_box
from pivot.planning.motion import MotionOptions, MotionPlan, MotionStatus, plan_motion
from pivot.planning.placement import check_placement, opening_warnings, propose_placements
from pivot.planning.rearrange import plan_rearrangement
from pivot.planning.snap import nearest_feasible_pose
from pivot.scene.graph import Scene, SceneObject, carried_objects, support_parents

JsonDict = dict[str, Any]


@dataclass
class Session:
    """The scene an agent works on, with its history.

    Attributes:
        scene: Current scene.
        history: Previous scenes, most recent last, for ``undo``.
        motions: Every executed motion, in order, for replay in the viewer.
        motion_counts: Number of motions before each history entry: one
            ``undo`` removes every motion of the move it reverts.
    """

    scene: Scene
    history: list[Scene] = field(default_factory=list)
    motions: list[MotionPlan] = field(default_factory=list)
    motion_counts: list[int] = field(default_factory=list)

    def record(self, scene: Scene, motions: list[MotionPlan]) -> None:
        """Executes a verified move: keeps the current scene for ``undo``.

        Args:
            scene: The scene after the move.
            motions: The motions that lead to it.
        """
        self.history.append(self.scene)
        self.motion_counts.append(len(self.motions))
        self.scene = scene
        self.motions.extend(motions)


def _round(value: float, digits: int = 3) -> float:
    """Rounds a float for compact JSON.

    Args:
        value: Value to round.
        digits: Number of decimals.

    Returns:
        The rounded value.
    """
    return round(float(value), digits)


def describe_object(scene: Scene, obj: SceneObject) -> JsonDict:
    """Summarizes an object for the model.

    Args:
        scene: The scene, to resolve what the object rests on.
        obj: The object.

    Returns:
        Its id, label, pose, size, support relations and how many people
        it takes to move it.
    """
    parents = support_parents(scene)
    box = obj.box
    return {
        "id": obj.object_id,
        "label": obj.label,
        "center_m": [_round(v) for v in box.center],
        "size_m": [_round(v) for v in box.size],
        "yaw_deg": _round(math.degrees(box.yaw), 1),
        "movable": obj.movable,
        "resting_on": parents.get(obj.object_id),
        "carries": carried_objects(scene, obj.object_id),
        "people_needed": people_needed(obj),
        "can_be_carried": can_be_carried(obj),
    }


def _pose(session: Session, object_id: int, x: float, y: float,
          yaw_deg: float | None) -> OrientedBox:
    """Builds the box of an object at a requested floor pose.

    Args:
        session: The session.
        object_id: The object.
        x: Target x of its center, in meters.
        y: Target y of its center, in meters.
        yaw_deg: Target yaw in degrees; unchanged if None.

    Returns:
        The box at the target pose.
    """
    box = session.scene[object_id].box
    yaw = box.yaw if yaw_deg is None else math.radians(yaw_deg)
    return box.moved(np.array([x, y], dtype=np.float64), yaw)


def list_objects(session: Session) -> JsonDict:
    """Lists every object, the extent of the walkable floor and the ceiling height.

    Args:
        session: The session.

    Returns:
        The objects, the floor bounding box and, if known, the ceiling height.
    """
    scene = session.scene
    result: JsonDict = {"objects": [describe_object(scene, o) for o in
                                    sorted(scene.objects.values(), key=lambda o: o.object_id)]}
    if scene.floor is not None:
        rows, cols = np.nonzero(scene.floor.floor)
        lo = scene.floor.origin + np.array([cols.min(), rows.min()]) * scene.floor.resolution
        hi = scene.floor.origin + (np.array([cols.max(), rows.max()]) + 1) * scene.floor.resolution
        result["floor_bounds_m"] = {"min_xy": [_round(v) for v in lo],
                                    "max_xy": [_round(v) for v in hi]}
    if scene.ceiling is not None:
        result["ceiling_height_m"] = _round(scene.ceiling)
    return result


def find_objects(session: Session, query: str) -> JsonDict:
    """Finds objects whose label contains a query.

    Args:
        session: The session.
        query: Substring of the label, e.g. ``"chair"``.

    Returns:
        The matching objects.
    """
    scene = session.scene
    return {"query": query, "matches": [describe_object(scene, o) for o in scene.find(query)]}


def measure_gap(session: Session, object_a: int, object_b: int) -> JsonDict:
    """Measures the distance between two objects.

    Args:
        session: The session.
        object_a: First object.
        object_b: Second object.

    Returns:
        The 3D gap and the gap between floor footprints.
    """
    a, b = session.scene[object_a].box, session.scene[object_b].box
    return {"object_a": object_a, "object_b": object_b,
            "gap_m": _round(box_distance(a, b)),
            "floor_gap_m": _round(polygon_distance(a.footprint(), b.footprint()))}


def check_pose(session: Session, object_id: int, x: float, y: float,
               yaw_deg: float | None = None, clearance_m: float = 0.0) -> JsonDict:
    """Checks whether an object could stand at a pose (destination only).

    Args:
        session: The session.
        object_id: The object.
        x: Target x of its center, in meters.
        y: Target y of its center, in meters.
        yaw_deg: Target yaw in degrees; unchanged if None.
        clearance_m: Required distance to other objects and walls.

    Returns:
        The placement report, with every conflict quantified.
    """
    box = _pose(session, object_id, x, y, yaw_deg)
    return check_placement(session.scene, object_id, box, clearance_m).to_dict()


def suggest_placements(session: Session, object_id: int, anchor_id: int,
                       max_gap_m: float = 0.4, clearance_m: float = 0.0,
                       max_results: int = 3) -> JsonDict:
    """Searches for free poses that put an object next to another one.

    Args:
        session: The session.
        object_id: The object to place.
        anchor_id: The object to place it next to.
        max_gap_m: Largest footprint gap that still counts as "next to".
        clearance_m: Required distance to other objects and walls.
        max_results: Maximum number of proposals.

    Returns:
        Feasible destinations, best first; they are not yet checked for reachability.
    """
    reports = propose_placements(session.scene, object_id, anchor_id, max_gap_m, clearance_m,
                                 max_results)
    return {"object_id": object_id, "anchor_id": anchor_id,
            "proposals": [r.to_dict() for r in reports],
            "note": "Destinations only: call plan_motion to check the object can get there."}


@dataclass(frozen=True)
class MoveRequest:
    """Arguments of the tools that plan or perform a move.

    Attributes:
        object_id: The object.
        x: Target x of its center, in meters.
        y: Target y of its center, in meters.
        yaw_deg: Target yaw in degrees; unchanged (but free to snap) if None.
        clearance_m: Hard minimum distance to obstacles and walls along the way.
        comfort_m: Extra margin the planner tries to keep; 0 for the least effort.
        snap_m: If the target is infeasible, use the nearest feasible pose
            within this distance instead; 0 disables it.
        carry_height_m: Carry the object with its bottom at this height, over
            lower furniture; it can then be put down on a surface. None
            slides it along the floor (unless its height must change).
        auto_carry: Without a carrying height, decide from the object: slide
            it, or carry it over the furniture when sliding is blocked or its
            height changes, if it can be carried (see :func:`can_be_carried`).
    """

    object_id: int
    x: float
    y: float
    yaw_deg: float | None = None
    clearance_m: float = 0.0
    comfort_m: float = 0.08
    snap_m: float = 0.0
    carry_height_m: float | None = None
    auto_carry: bool = False

    @property
    def options(self) -> MotionOptions:
        """Motion parameters of the request."""
        return MotionOptions(clearance=self.clearance_m, comfort=self.comfort_m)


def _goal(session: Session, request: MoveRequest) -> tuple[OrientedBox, JsonDict]:
    """Builds the goal pose of a move: resting on a surface, snapped if asked.

    Args:
        session: The session.
        request: The move.

    Returns:
        The goal, and notes for the response: the adjustment, if any, and
        warnings about the goal (in front of a window).
    """
    object_id = request.object_id
    current = session.scene[object_id].box
    # Surfaces up to the carrying height (or the current one) can receive the object.
    below = max(current.z_min, request.carry_height_m or _auto_height(session, request) or 0.0)
    goal = resting_box(session.scene, object_id,
                       _pose(session, object_id, request.x, request.y, request.yaw_deg), below)
    snapped = (nearest_feasible_pose(session.scene, object_id, goal, request.snap_m,
                                     request.clearance_m, allow_rotation=request.yaw_deg is None)
               if request.snap_m > 0.0 else None)
    if snapped is None or snapped is goal:
        return goal, _warnings(session, goal)
    snapped = resting_box(session.scene, object_id, snapped, below)
    return snapped, {**_warnings(session, snapped), "goal_adjusted": {
        "x": _round(snapped.center[0]), "y": _round(snapped.center[1]),
        "yaw_deg": _round(math.degrees(snapped.yaw), 1),
        "shift_m": _round(float(np.linalg.norm(snapped.center[:2] - goal.center[:2]))),
        "reason": "The requested pose was infeasible; this is the nearest feasible one."}}


def _auto_height(session: Session, request: MoveRequest) -> float | None:
    """Gives the carrying height chosen automatically, for objects that can be carried.

    Args:
        session: The session.
        request: The move.

    Returns:
        The height, or None if the request does not ask for it or the object
        cannot be carried.
    """
    if not request.auto_carry or request.carry_height_m is not None:
        return None
    if not can_be_carried(session.scene[request.object_id]):
        return None
    return carry_height(session.scene, request.object_id)


def _auto_plan(session: Session, request: MoveRequest, goal: OrientedBox) -> MotionPlan:
    """Slides an object if it can, else carries it over the furniture if it can be carried.

    Args:
        session: The session.
        request: The move, with ``auto_carry``.
        goal: Its goal pose, resting on its support.

    Returns:
        The plan.
    """
    obj = session.scene[request.object_id]
    start = obj.box
    slid = None
    if abs(goal.z_min - start.z_min) < 1e-3:
        slid = plan_motion(session.scene, request.object_id, goal, request.options)
        if slid.status is MotionStatus.SUCCESS:
            return slid
    height = _auto_height(session, request)
    if height is None:
        return slid or MotionPlan(
            MotionStatus.INVALID_GOAL, request.object_id,
            message=f"The {obj.label} is too heavy or too tall to lift: it can only slide "
                    "along the floor.")
    carried = plan_carried_motion(session.scene, request.object_id, goal,
                                  max(height, start.z_min, goal.z_min), request.options)
    if carried.status is not MotionStatus.SUCCESS and slid is not None:
        return slid
    if slid is not None:
        return dataclasses.replace(carried, message=f"Carried over the furniture, as sliding "
                                                    f"was blocked. {carried.message}")
    return carried


def _plan(session: Session, request: MoveRequest, goal: OrientedBox) -> MotionPlan:
    """Plans a move along the floor, or a carried move when the height changes.

    A carry that fails (a chair under a desk cannot be lifted where it
    stands) falls back to sliding, when the object stays at the same height.

    Args:
        session: The session.
        request: The move.
        goal: Its goal pose, resting on its support.

    Returns:
        The plan.
    """
    if request.auto_carry and request.carry_height_m is None:
        return _auto_plan(session, request, goal)
    start = session.scene[request.object_id].box
    same_height = abs(goal.z_min - start.z_min) < 1e-3
    if request.carry_height_m is None and same_height:
        return plan_motion(session.scene, request.object_id, goal, request.options)
    carry = max(request.carry_height_m or 0.0, start.z_min, goal.z_min)
    carried = plan_carried_motion(session.scene, request.object_id, goal, carry,
                                  request.options)
    if carried.status is MotionStatus.SUCCESS or not same_height:
        return carried
    slid = plan_motion(session.scene, request.object_id, goal, request.options)
    if slid.status is not MotionStatus.SUCCESS:
        return carried
    return dataclasses.replace(slid, message=f"Slid along the floor instead of carried "
                                             f"({carried.message}) {slid.message}")


def _warnings(session: Session, goal: OrientedBox) -> JsonDict:
    """Lists remarks about a goal pose that do not make it infeasible.

    Args:
        session: The session.
        goal: The goal pose.

    Returns:
        ``{"warnings": [...]}``, or an empty dictionary.
    """
    warnings = opening_warnings(session.scene, goal)
    return {"warnings": warnings} if warnings else {}


def plan_path(session: Session, **arguments: Any) -> JsonDict:
    """Checks whether an object can be moved to a pose, and how.

    Args:
        session: The session.
        **arguments: Fields of :class:`MoveRequest`.

    Returns:
        The motion plan: status, path, effort and tightest clearance.
    """
    request = MoveRequest(**arguments)
    goal, notes = _goal(session, request)
    return {**_plan(session, request, goal).to_dict(), **notes}


def move_object(session: Session, **arguments: Any) -> JsonDict:
    """Moves an object, only if a collision-free motion exists.

    Args:
        session: The session.
        **arguments: Fields of :class:`MoveRequest`.

    Returns:
        Whether the move happened, the motion plan, and what moved along.
    """
    request = MoveRequest(**arguments)
    goal, notes = _goal(session, request)
    plan = _plan(session, request, goal)
    result: JsonDict = {"moved": plan.status is MotionStatus.SUCCESS, "plan": plan.to_dict(),
                        **notes}
    if plan.status is MotionStatus.SUCCESS:
        result["carried_along"] = carried_objects(session.scene, request.object_id)
        session.record(session.scene.moved(request.object_id, goal), [plan])
    return result


def make_room(session: Session, **arguments: Any) -> JsonDict:
    """Checks whether an object can reach a pose once blocking objects are moved aside.

    Args:
        session: The session.
        **arguments: Fields of :class:`MoveRequest`.

    Returns:
        The verified steps (blockers first, the object last) or the reason
        none were found. Nothing moves.
    """
    request = MoveRequest(**arguments)
    goal, notes = _goal(session, request)
    return {**plan_rearrangement(session.scene, request.object_id, goal,
                                 request.options).to_dict(), **notes}


def make_room_and_move(session: Session, **arguments: Any) -> JsonDict:
    """Moves blocking objects aside, then the object; only if every step is verified.

    Args:
        session: The session.
        **arguments: Fields of :class:`MoveRequest`.

    Returns:
        Whether the moves happened, and each step with what moved along. One
        ``undo`` reverts all of them.
    """
    request = MoveRequest(**arguments)
    goal, notes = _goal(session, request)
    plan = plan_rearrangement(session.scene, request.object_id, goal, request.options)
    result: JsonDict = {"moved": plan.success, **plan.to_dict(), **notes}
    if plan.success:
        scene = session.scene
        for step, data in zip(plan.steps, result["steps"]):
            data["carried_along"] = carried_objects(scene, step.object_id)
            scene = scene.moved(step.object_id, step.waypoints[-1])
        session.record(scene, plan.steps)
    return result


def check_passages(session: Session, width_m: float = WALKING_WIDTH) -> JsonDict:
    """Checks that a person of a given width can still get around the room.

    Args:
        session: The session.
        width_m: Width needed to pass, in meters: 0.6 walking, 0.9 a wheelchair.

    Returns:
        The share of the free floor reachable from the doors, the doors with
        no room in front, and the objects nobody can get within reach of.
    """
    report = check_access(session.scene, width_m)
    if report is None:
        return {"error": "The floor of this scene is unknown: passages cannot be checked."}
    return report.to_dict()


def undo(session: Session) -> JsonDict:
    """Reverts the last executed move.

    Args:
        session: The session.

    Returns:
        Whether a move was undone.
    """
    if not session.history:
        return {"undone": False, "reason": "Nothing to undo."}
    session.scene = session.history.pop()
    del session.motions[session.motion_counts.pop():]
    return {"undone": True}


@dataclass(frozen=True)
class Tool:
    """A function callable by the model.

    Attributes:
        name: Function name seen by the model.
        description: When and why to use it.
        parameters: JSON schema of the arguments.
        handler: Implementation, called with the session and the arguments.
    """

    name: str
    description: str
    parameters: JsonDict
    handler: Callable[..., JsonDict]

    def schema(self) -> JsonDict:
        """Returns the tool definition in the Chat Completions format.

        Returns:
            The ``{"type": "function", ...}`` definition.
        """
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


def _object_schema(properties: dict[str, JsonDict], required: list[str]) -> JsonDict:
    """Builds a JSON schema for an object with the given properties.

    Args:
        properties: Property schemas.
        required: Required property names.

    Returns:
        The schema.
    """
    return {"type": "object", "properties": properties, "required": required,
            "additionalProperties": False}


_ID = {"type": "integer", "description": "Object id, from list_objects."}
_POSE = {
    "object_id": _ID,
    "x": {"type": "number", "description": "Target x of the object's center, meters."},
    "y": {"type": "number", "description": "Target y of the object's center, meters."},
    "yaw_deg": {"type": "number", "description": "Target orientation, degrees (default: keep)."},
    "clearance_m": {"type": "number", "description": "Required gap to obstacles and walls."},
}
_COMFORT = {"type": "number", "description": "Extra margin to keep when possible (default 0.08 "
                                             "m); 0 gives the least-effort path."}
_CARRY = {"type": "number", "description": "Carry the object (only if can_be_carried) with its "
                                           "bottom at this height in meters, over lower "
                                           "furniture; it can then be put down on a table or a "
                                           "shelf. Default: slide it along the floor."}
_SNAP = {"type": "number", "description": "If the target is infeasible (e.g. too close to a wall), "
                                          "use the nearest feasible pose within this many meters "
                                          "(default 0: no adjustment)."}

TOOLS = [
    Tool("list_objects", "List every object (id, label, center, size, yaw, support relations, "
         "people needed to move it), the bounds of the walkable floor and the ceiling height, "
         "if known. Call this first.",
         _object_schema({}, []), list_objects),
    Tool("find_objects", "Find objects whose label contains a word.",
         _object_schema({"query": {"type": "string"}}, ["query"]), find_objects),
    Tool("measure_gap", "Measure the distance between two objects.",
         _object_schema({"object_a": _ID, "object_b": _ID}, ["object_a", "object_b"]),
         measure_gap),
    Tool("check_placement", "Check whether an object could stand at a pose: collisions, walls, "
         "floor. Says nothing about whether it can get there.",
         _object_schema(_POSE, ["object_id", "x", "y"]), check_pose),
    Tool("propose_placements", "Find free destinations next to another object.",
         _object_schema({"object_id": _ID, "anchor_id": _ID,
                         "max_gap_m": {"type": "number"}, "clearance_m": {"type": "number"},
                         "max_results": {"type": "integer"}}, ["object_id", "anchor_id"]),
         suggest_placements),
    Tool("plan_motion", "Check whether an object can be moved to a pose without hitting "
         "anything on the way, and return the path, effort and tightest clearance.",
         _object_schema({**_POSE, "comfort_m": _COMFORT, "snap_m": _SNAP,
                         "carry_height_m": _CARRY}, ["object_id", "x", "y"]), plan_path),
    Tool("move_object", "Move an object to a pose. Only happens if a collision-free motion "
         "exists; objects resting on it move along.",
         _object_schema({**_POSE, "comfort_m": _COMFORT, "snap_m": _SNAP,
                         "carry_height_m": _CARRY}, ["object_id", "x", "y"]), move_object),
    Tool("make_room", "When an object cannot reach a pose, check whether moving other objects "
         "out of the way (e.g. chairs blocking a passage) makes it possible. Nothing moves.",
         _object_schema({k: _POSE[k] for k in ("object_id", "x", "y", "yaw_deg")}
                        | {"snap_m": _SNAP}, ["object_id", "x", "y"]), make_room),
    Tool("make_room_and_move", "Move the objects in the way aside, then the object. Only "
         "happens if every step is a verified collision-free motion; one undo reverts all.",
         _object_schema({k: _POSE[k] for k in ("object_id", "x", "y", "yaw_deg")}
                        | {"snap_m": _SNAP}, ["object_id", "x", "y"]), make_room_and_move),
    Tool("check_access", "Check that a person can still get around the room from the doors: "
         "reachable floor, blocked doors, objects nobody can reach. width_m: 0.6 walking "
         "(default), 0.9 a wheelchair.",
         _object_schema({"width_m": {"type": "number"}}, []), check_passages),
    Tool("undo", "Revert the last move.", _object_schema({}, []), undo),
]
TOOLS_BY_NAME = {tool.name: tool for tool in TOOLS}


def call_tool(session: Session, name: str, arguments: str) -> str:
    """Executes a tool call and serializes its result.

    Errors (unknown tool, bad arguments, unknown object) are returned to the
    model as JSON so it can correct itself.

    Args:
        session: The session.
        name: Tool name.
        arguments: JSON-encoded arguments.

    Returns:
        The JSON-encoded result.
    """
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return json.dumps({"error": f"Unknown tool {name!r}."})
    try:
        kwargs = json.loads(arguments or "{}")
        return json.dumps(tool.handler(session, **kwargs))
    except (json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
        return json.dumps({"error": f"{type(error).__name__}: {error}"})
