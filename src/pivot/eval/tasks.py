"""Rearrangement tasks with checkable success criteria.

Each task is a request in words ("Move the chair (#5) next to the table
(#72)") together with the facts needed to judge any answer to it: which
object must move, the predicate its final pose must satisfy, and whether a
solution is known to exist. Solutions are searched with the geometric tools,
generously; a task with none found is labeled infeasible, and the right
answer to it is a refusal. Labels are lower bounds: an inventive answer
(moving another object first) may solve a task labeled infeasible.

Besides simple placements, some kinds need real geometry: pushing a chair
under a table (its backrest may not pass under the top), putting an object
on a table (it must be carried), reaching a spot blocked by other objects
(they must be moved aside first), taking an object into another room
(through the doorways), turning it where it stands, clearing a door it
blocks, putting it against a wall, and making way for a wheelchair to a
table (a clear passage and a free spot, judged on the room as it ends up).
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from enum import Enum
from typing import Any

import numpy as np

from pivot.agent.tools import (
    Session, make_room_and_move, make_way, move_object, move_to_room, plan_path,
)
from pivot.geometry.collision import footprint_overlap, polygon_distance
from pivot.planning.placement import check_placement, opening_warnings, propose_placements
from pivot.planning.way import way_is_clear
from pivot.scene.graph import Scene, SceneObject, is_flat, is_opening, is_supported_by
from pivot.scene.rooms import find_rooms

# Footprint gap under which an object is "next to" another, in meters.
NEXT_TO_GAP = 0.3
# Distance under which a moved object is "at" a requested point, in meters.
AT_POINT_TOLERANCE = 0.15
# Fraction of a chair's footprint that must lie under a table for it to be "under" it.
TUCK_FRACTION = 0.3
# The criterion is stated: "as far as it goes" alone would make any push a success.
TUCK_PROMPT = "Push {mover} under {table}, with at least a third of it under the table top."
# Bottom height of carried objects, above desks, in meters.
CARRY_HEIGHT = 0.9
# Error tolerated on a requested turn, in degrees.
TURN_TOLERANCE_DEG = 15.0
# Draws of a task before giving up on one not already in the set.
_DRAWS = 5
# Gap to a wall under which an object stands "against" it, in meters.
WALL_GAP = 0.1


class TaskKind(str, Enum):
    """What a task asks for."""

    NEXT_TO = "next_to"
    NEXT_TO_CLEAR = "next_to_clear"
    AT_POINT = "at_point"
    TUCK_UNDER = "tuck_under"
    ON_SURFACE = "on_surface"
    AT_POINT_BLOCKED = "at_point_blocked"
    TO_ROOM = "to_room"
    TURN = "turn_in_place"
    CLEAR_DOOR = "clear_door"
    AGAINST_WALL = "against_wall"
    WHEELCHAIR = "wheelchair_way"


# Kinds judged on the room as it ends up rather than on one object's pose: the
# "object" of the task is then the one to reach, which need not move.
ROOM_KINDS = frozenset({TaskKind.WHEELCHAIR})


@dataclass(frozen=True)
class Task:
    """One request and what a correct answer must achieve.

    Attributes:
        task_id: Unique name.
        kind: What is asked.
        prompt: The request, in English.
        object_id: The object to move.
        anchor_id: The object to move it next to, under or onto.
        point: Requested center ``(x, y)``, for ``AT_POINT*`` tasks; the
            center to keep, for ``TURN``.
        feasible: Whether a solution was found when the task was made.
        room: The room to go to, for ``TO_ROOM``.
        yaw_deg: The orientation to end with, for ``TURN``, in degrees.
    """

    task_id: str
    kind: TaskKind
    prompt: str
    object_id: int
    anchor_id: int | None = None
    point: tuple[float, float] | None = None
    feasible: bool = True
    room: str | None = None
    yaw_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the task.

        Returns:
            A JSON-compatible dictionary.
        """
        return {**asdict(self), "kind": self.kind.value}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Reads a task written by :meth:`to_dict`.

        Args:
            data: The dictionary.

        Returns:
            The task.
        """
        point = data.get("point")
        return cls(**{**data, "kind": TaskKind(data["kind"]),
                      "point": tuple(point) if point is not None else None})


def satisfies(scene: Scene, task: Task, obj: SceneObject) -> bool:
    """Tests the task's predicate on an object's final pose.

    Reachability is not tested here, only where the object ends up.

    Args:
        scene: The final scene.
        task: The task.
        obj: The moved object, at its final pose.

    Returns:
        True if the pose is what the task asks for.
    """
    if task.kind in (TaskKind.AT_POINT, TaskKind.AT_POINT_BLOCKED):
        assert task.point is not None
        return bool(np.linalg.norm(obj.box.center[:2] - np.array(task.point)) <= AT_POINT_TOLERANCE)
    other = _satisfies_new_kind(scene, task, obj)
    if other is not None:
        return other
    assert task.anchor_id is not None
    anchor = scene[task.anchor_id].box
    if task.kind is TaskKind.TUCK_UNDER:
        return obj.box.z_min < 0.05 and footprint_overlap(obj.box, anchor) >= TUCK_FRACTION
    if task.kind is TaskKind.ON_SURFACE:
        return is_supported_by(obj.box, anchor)
    if polygon_distance(obj.box.footprint(), anchor.footprint()) > NEXT_TO_GAP:
        return False
    return task.kind is TaskKind.NEXT_TO or not opening_warnings(scene, obj.box)


def _satisfies_new_kind(scene: Scene, task: Task, obj: SceneObject) -> bool | None:
    """Tests the predicates of the kinds beyond placements next to, under or onto.

    Args:
        scene: The final scene.
        task: The task.
        obj: The task's object, at its final pose.

    Returns:
        Whether the predicate holds; None for the other kinds.
    """
    if task.kind is TaskKind.TO_ROOM:
        rooms = find_rooms(scene)
        target = rooms.find(task.room or "") if rooms is not None else None
        return rooms is not None and target is not None \
            and rooms.room_at(obj.box.center[:2]) == target.room_id
    if task.kind is TaskKind.TURN:
        assert task.point is not None and task.yaw_deg is not None
        turn = math.degrees(math.remainder(obj.box.yaw - math.radians(task.yaw_deg), math.pi))
        return bool(np.linalg.norm(obj.box.center[:2] - np.array(task.point))
                    <= AT_POINT_TOLERANCE and abs(turn) <= TURN_TOLERANCE_DEG)
    if task.kind is TaskKind.CLEAR_DOOR:
        return not any("door" in w for w in opening_warnings(scene, obj.box))
    if task.kind is TaskKind.AGAINST_WALL:
        return scene.floor is not None and scene.floor.distance_to_walls(
            obj.box.footprint(), search_radius=2 * WALL_GAP) <= WALL_GAP
    if task.kind is TaskKind.WHEELCHAIR:
        return way_is_clear(scene, obj.object_id)
    return None


def solve(scene: Scene, task: Task, max_checks: int = 12) -> Session | None:
    """Solves a task with the geometric tools, the way a careful planner would.

    The request's structured fields are used, not its words: a constraint
    stated only in words is ignored unless it is part of the task's predicate.

    Args:
        scene: The initial scene.
        task: The task.
        max_checks: Maximum number of placements tried next to an anchor.

    Returns:
        The session after the verified moves (its ``motions`` count the moves
        made), or None if no solution was found.
    """
    session = Session(scene)
    object_id = task.object_id
    if task.kind in (TaskKind.TO_ROOM, TaskKind.TURN, TaskKind.CLEAR_DOOR,
                     TaskKind.AGAINST_WALL, TaskKind.WHEELCHAIR):
        return _solve_new_kind(session, task, max_checks)
    if task.kind in (TaskKind.NEXT_TO, TaskKind.NEXT_TO_CLEAR):
        assert task.anchor_id is not None
        reports = propose_placements(scene, object_id, task.anchor_id, NEXT_TO_GAP,
                                     max_results=4 * max_checks)
        candidates = [r.box for r in reports
                      if satisfies(scene, task, dataclasses.replace(scene[object_id], box=r.box))]
        for box in candidates[:max_checks]:
            if move_object(session, object_id=object_id, x=float(box.center[0]),
                           y=float(box.center[1]),
                           yaw_deg=math.degrees(box.yaw))["moved"]:
                return session
        return None
    if task.kind in (TaskKind.AT_POINT, TaskKind.AT_POINT_BLOCKED):
        assert task.point is not None
        arguments: dict[str, Any] = {"object_id": object_id, "x": task.point[0],
                                     "y": task.point[1]}
        moved = move_object(session, **arguments)["moved"] or (
            task.kind is TaskKind.AT_POINT_BLOCKED
            and make_room_and_move(session, **arguments)["moved"])
        return session if moved else None
    assert task.anchor_id is not None
    center = scene[task.anchor_id].box.center
    arguments = {"object_id": object_id, "x": float(center[0]), "y": float(center[1]),
                 "snap_m": 0.8}
    if task.kind is TaskKind.ON_SURFACE:
        arguments |= {"carry_height_m": CARRY_HEIGHT, "snap_m": 0.4}
    if not move_object(session, **arguments)["moved"]:
        return None
    return session if satisfies(session.scene, task, session.scene[object_id]) else None


def _solve_new_kind(session: Session, task: Task, max_checks: int) -> Session | None:
    """Solves the kinds beyond placements next to, under or onto, with the tools.

    Args:
        session: A session on the initial scene.
        task: The task.
        max_checks: Maximum number of spots tried.

    Returns:
        The session after the verified moves, or None if no solution was found.
    """
    scene, object_id = session.scene, task.object_id
    if task.kind is TaskKind.TO_ROOM:
        moved = move_to_room(session, object_id, task.room or "")
        return session if moved.get("moved") else None
    if task.kind is TaskKind.WHEELCHAIR:
        return session if make_way(session, object_id)["success"] else None
    if task.kind is TaskKind.TURN:
        assert task.point is not None and task.yaw_deg is not None
        moved = move_object(session, object_id=object_id, x=task.point[0], y=task.point[1],
                            yaw_deg=task.yaw_deg)
        return session if moved["moved"] else None
    for box in _spots_for(scene, task)[:max_checks]:
        if move_object(session, object_id=object_id, x=float(box.center[0]),
                       y=float(box.center[1]), yaw_deg=math.degrees(box.yaw))["moved"]:
            return session
    return None


def _spots_for(scene: Scene, task: Task) -> list[Any]:
    """Lists free poses of the object meeting a ``CLEAR_DOOR`` or ``AGAINST_WALL`` task.

    Args:
        scene: The initial scene.
        task: The task.

    Returns:
        Boxes, the nearest to the object first.
    """
    obj = scene[task.object_id]
    current = obj.box
    spots = []
    for distance in np.arange(0.3, 3.01, 0.15):
        for angle in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False):
            for turn in (0.0, 0.5 * math.pi):
                box = current.moved(current.center[:2] + distance * np.array(
                    [math.cos(angle), math.sin(angle)]), current.yaw + turn)
                moved = dataclasses.replace(obj, box=box)
                if satisfies(scene, task, moved) and check_placement(scene, obj.object_id,
                                                                     box).feasible:
                    spots.append(box)
        if len(spots) >= 24:
            break
    return spots


def _floor_objects(scene: Scene, max_area: float = 1.0) -> list[SceneObject]:
    """Lists the movable objects standing on the floor.

    Args:
        scene: The scene.
        max_area: Largest footprint, in square meters.

    Returns:
        The objects, by id.
    """
    return sorted((o for o in scene.objects.values()
                   if o.movable and not is_opening(o) and o.box.z_min < 0.05
                   and o.box.size[0] * o.box.size[1] < max_area), key=lambda o: o.object_id)


def _name(obj: SceneObject) -> str:
    """Returns how a request names an object.

    Args:
        obj: The object.

    Returns:
        E.g. ``"the chair (#5)"``.
    """
    return f"the {obj.label} (#{obj.object_id})"


def generate_tasks(scene: Scene, count: int = 36, seed: int = 0) -> list[Task]:
    """Makes a mix of tasks on a scene and labels each with its feasibility.

    The kinds take turns: next to a table, the same keeping doors and windows
    clear, at a precise position (half on free floor, half on top of other
    furniture, which mostly cannot work; large furniture included), a chair
    under a table, an object onto a table, a position whose way is blocked by
    other objects, another room, a quarter turn in place, a door to clear,
    against a wall, and a wheelchair's way to a table. A kind the scene has
    nothing for (one room only) is skipped, and so is a task drawn again.

    Args:
        scene: The scene.
        count: Number of tasks.
        seed: Random seed.

    Returns:
        The tasks.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    prompts: set[str] = set()
    for index in range(count):
        kind = list(TaskKind)[index % len(TaskKind)]
        for _ in range(_DRAWS):
            task = _make_task(scene, rng, f"t{index:03d}", kind,
                              on_furniture=(index // 6) % 2 == 0)
            if task is None or task.prompt not in prompts:
                break
        # The scene may have no object for this kind, or no other task of it.
        if task is not None and task.prompt not in prompts:
            prompts.add(task.prompt)
            tasks.append(dataclasses.replace(task, feasible=solve(scene, task) is not None))
    return tasks


def _make_task(scene: Scene, rng: np.random.Generator, task_id: str, kind: TaskKind,
               on_furniture: bool) -> Task | None:
    """Draws one task of a given kind.

    Args:
        scene: The scene.
        rng: Random generator.
        task_id: Name of the task.
        kind: Its kind.
        on_furniture: For ``AT_POINT`` tasks, whether to aim at other furniture.

    Returns:
        The task, not yet labeled; None if the scene has no object for it.
    """
    if kind is TaskKind.AT_POINT_BLOCKED:
        return _blocked_task(scene, rng, task_id)
    if kind is TaskKind.AT_POINT:
        mover = _pick(rng, _floor_objects(scene, max_area=1.6))
        x, y = _target_point(scene, rng, on_furniture)
        return Task(task_id, kind, f"Move {_name(mover)} so that its center is at "
                    f"x = {x} m, y = {y} m.", mover.object_id, point=(x, y))
    if kind in (TaskKind.TUCK_UNDER, TaskKind.ON_SURFACE):
        return _table_task(scene, rng, task_id, kind)
    if kind in (TaskKind.TO_ROOM, TaskKind.TURN, TaskKind.CLEAR_DOOR, TaskKind.AGAINST_WALL,
                TaskKind.WHEELCHAIR):
        return _new_kind_task(scene, rng, task_id, kind)
    mover = _pick(rng, _floor_objects(scene))
    anchors = [o for o in scene.objects.values()
               if o.label in ("table", "desk", "cabinet", "shelf") and o.box.z_min < 0.05
               and o.object_id != mover.object_id]
    if not anchors:
        return None
    anchor = _pick(rng, anchors)
    clear = " Keep doors and windows clear." if kind is TaskKind.NEXT_TO_CLEAR else ""
    return Task(task_id, kind, f"Move {_name(mover)} next to {_name(anchor)}." + clear,
                mover.object_id, anchor_id=anchor.object_id)


def _wheelchair_task(scene: Scene, rng: np.random.Generator, task_id: str) -> Task | None:
    """Draws a task of making room for a wheelchair at a table.

    Args:
        scene: The scene.
        rng: Random generator.
        task_id: Name of the task.

    Returns:
        The task; None if the scene has no table.
    """
    tables = [o for o in scene.objects.values() if o.label.lower() in ("table", "desk")]
    if not tables:
        return None
    table = _pick(rng, tables)
    return Task(task_id, TaskKind.WHEELCHAIR,
                f"Make room for a person in a wheelchair at {_name(table)}: "
                "a clear passage 90 cm wide from a door, and a free spot at the table.",
                table.object_id, anchor_id=table.object_id)


def _to_room_task(scene: Scene, rng: np.random.Generator, task_id: str) -> Task | None:
    """Draws a task of moving an object into another room.

    Args:
        scene: The scene.
        rng: Random generator.
        task_id: Name of the task.

    Returns:
        The task; None if the scene has one room only.
    """
    rooms = find_rooms(scene)
    if rooms is None or len(rooms.rooms) < 2:
        return None
    rugs = [o for o in scene.objects.values() if is_flat(o) and o.movable]
    mover = _pick(rng, _floor_objects(scene) + rugs)
    here = rooms.room_at(mover.box.center[:2])
    room = _pick_room(rng, [r for r in rooms.rooms if r.room_id != here])
    return Task(task_id, TaskKind.TO_ROOM, f"Move {_name(mover)} into the {room}.",
                mover.object_id, room=room)


def _new_kind_task(scene: Scene, rng: np.random.Generator, task_id: str,
                   kind: TaskKind) -> Task | None:
    """Draws a task of the kinds beyond placements next to, under or onto.

    Args:
        scene: The scene.
        rng: Random generator.
        task_id: Name of the task.
        kind: Its kind.

    Returns:
        The task, not yet labeled; None if the scene has nothing for it (one
        room only, no object in front of a door, no table).
    """
    if kind is TaskKind.WHEELCHAIR:
        return _wheelchair_task(scene, rng, task_id)
    if kind is TaskKind.TO_ROOM:
        return _to_room_task(scene, rng, task_id)
    movers = _floor_objects(scene)
    if kind is TaskKind.CLEAR_DOOR:
        movers = [o for o in movers
                  if any("door" in w for w in opening_warnings(scene, o.box))]
        if not movers:
            return None
        mover = _pick(rng, movers)
        return Task(task_id, kind, f"{_name(mover).capitalize()} stands in front of a door: "
                    "move it so that it no longer blocks any door.", mover.object_id)
    mover = _pick(rng, movers)
    if kind is TaskKind.TURN:
        yaw = round(math.degrees(mover.box.yaw) + 90.0, 1)
        center = (round(float(mover.box.center[0]), 2), round(float(mover.box.center[1]), 2))
        return Task(task_id, kind, f"Turn {_name(mover)} a quarter turn (90 degrees) where it "
                    "stands.", mover.object_id, point=center, yaw_deg=yaw)
    return Task(task_id, kind, f"Put {_name(mover)} against a wall.", mover.object_id)


def _pick_room(rng: np.random.Generator, rooms: list[Any]) -> str:
    """Draws the name of a room.

    Args:
        rng: Random generator.
        rooms: Candidates; not empty.

    Returns:
        The room's name.
    """
    return str(rooms[rng.integers(len(rooms))].name)


def _table_task(scene: Scene, rng: np.random.Generator, task_id: str,
                kind: TaskKind) -> Task | None:
    """Draws a chair to push under a table, or a light object to put on one.

    Args:
        scene: The scene.
        rng: Random generator.
        task_id: Name of the task.
        kind: ``TUCK_UNDER`` or ``ON_SURFACE``.

    Returns:
        The task; None if the scene has no suitable object.
    """
    table = _pick(rng, [o for o in scene.objects.values()
                        if o.label.lower() in ("table", "desk")])
    if kind is TaskKind.TUCK_UNDER:
        movers = [o for o in scene.objects.values() if "chair" in o.label.lower()]
        prompt = TUCK_PROMPT
    else:
        movers = [o for o in scene.objects.values() if o.movable and not is_opening(o)
                  and o.box.size[2] <= 0.9 and o.box.size[0] * o.box.size[1] <= 0.4
                  and not is_supported_by(o.box, table.box)]
        prompt = "Put {mover} on {table}."
    if not movers:
        return None
    mover = _pick(rng, movers)
    return Task(task_id, kind, prompt.format(mover=_name(mover), table=_name(table)),
                mover.object_id, anchor_id=table.object_id)


def _blocked_task(scene: Scene, rng: np.random.Generator, task_id: str,
                  tries: int = 25) -> Task:
    """Draws a position that is free but out of reach while other objects stand in the way.

    Args:
        scene: The scene.
        rng: Random generator.
        task_id: Name of the task.
        tries: Maximum number of (object, position) pairs drawn.

    Returns:
        The task; the last pair drawn if none was blocked.
    """
    movers = _floor_objects(scene, max_area=1.6)
    for _ in range(tries):
        mover = _pick(rng, movers)
        x, y = _target_point(scene, rng, on_furniture=False)
        if plan_path(Session(scene), object_id=mover.object_id, x=x, y=y)["status"] == "no_path":
            break
    return Task(task_id, TaskKind.AT_POINT_BLOCKED, f"Move {_name(mover)} so that its center is "
                f"at x = {x} m, y = {y} m. Other objects may be moved out of the way.",
                mover.object_id, point=(x, y))


def _pick(rng: np.random.Generator, items: list[SceneObject]) -> SceneObject:
    """Draws one object.

    Args:
        rng: Random generator.
        items: Candidates; not empty.

    Returns:
        The object.
    """
    return items[rng.integers(len(items))]


def _target_point(scene: Scene, rng: np.random.Generator,
                  on_furniture: bool) -> tuple[float, float]:
    """Draws a requested position: a free floor cell, or the center of some furniture.

    Args:
        scene: The scene, with a floor map.
        rng: Random generator.
        on_furniture: Whether to aim at another object.

    Returns:
        ``(x, y)``, rounded to the centimeter.
    """
    if on_furniture:
        others = [o for o in scene.objects.values() if not is_opening(o)]
        point = others[rng.integers(len(others))].box.center[:2]
    else:
        assert scene.floor is not None
        rows, cols = np.nonzero(scene.floor.floor)
        cell = rng.integers(len(rows))
        point = scene.floor.origin + (np.array([cols[cell], rows[cell]]) + 0.5) \
            * scene.floor.resolution
    return round(float(point[0]), 2), round(float(point[1]), 2)


def save_tasks(tasks: list[Task], path: Path) -> None:
    """Writes tasks as JSON.

    Args:
        tasks: The tasks.
        path: Output file.
    """
    path.write_text(json.dumps([t.to_dict() for t in tasks], indent=1), encoding="utf-8")


def load_tasks(path: Path) -> list[Task]:
    """Reads tasks written by :func:`save_tasks`.

    Args:
        path: Input file.

    Returns:
        The tasks.
    """
    return [Task.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]
