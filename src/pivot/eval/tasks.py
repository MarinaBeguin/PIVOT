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
(they must be moved aside first).
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

from pivot.agent.tools import Session, make_room_and_move, move_object, plan_path
from pivot.geometry.collision import footprint_overlap, polygon_distance
from pivot.planning.placement import opening_warnings, propose_placements
from pivot.scene.graph import Scene, SceneObject, is_opening, is_supported_by

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


class TaskKind(str, Enum):
    """What a task asks for."""

    NEXT_TO = "next_to"
    NEXT_TO_CLEAR = "next_to_clear"
    AT_POINT = "at_point"
    TUCK_UNDER = "tuck_under"
    ON_SURFACE = "on_surface"
    AT_POINT_BLOCKED = "at_point_blocked"


@dataclass(frozen=True)
class Task:
    """One request and what a correct answer must achieve.

    Attributes:
        task_id: Unique name.
        kind: What is asked.
        prompt: The request, in English.
        object_id: The object to move.
        anchor_id: The object to move it next to, under or onto.
        point: Requested center ``(x, y)``, for ``AT_POINT*`` tasks.
        feasible: Whether a solution was found when the task was made.
    """

    task_id: str
    kind: TaskKind
    prompt: str
    object_id: int
    anchor_id: int | None = None
    point: tuple[float, float] | None = None
    feasible: bool = True

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
    assert task.anchor_id is not None
    anchor = scene[task.anchor_id].box
    if task.kind is TaskKind.TUCK_UNDER:
        return obj.box.z_min < 0.05 and footprint_overlap(obj.box, anchor) >= TUCK_FRACTION
    if task.kind is TaskKind.ON_SURFACE:
        return is_supported_by(obj.box, anchor)
    if polygon_distance(obj.box.footprint(), anchor.footprint()) > NEXT_TO_GAP:
        return False
    return task.kind is TaskKind.NEXT_TO or not opening_warnings(scene, obj.box)


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
    under a table, an object onto a table, and a position whose way is
    blocked by other objects.

    Args:
        scene: The scene.
        count: Number of tasks.
        seed: Random seed.

    Returns:
        The tasks.
    """
    rng = np.random.default_rng(seed)
    tasks = []
    for index in range(count):
        kind = list(TaskKind)[index % len(TaskKind)]
        task = _make_task(scene, rng, f"t{index:03d}", kind, on_furniture=(index // 6) % 2 == 0)
        if task is not None:  # The scene may have no object for this kind.
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
    mover = _pick(rng, _floor_objects(scene))
    anchors = [o for o in scene.objects.values()
               if o.label in ("table", "desk", "cabinet", "shelf") and o.box.z_min < 0.05]
    anchor = _pick(rng, anchors)
    clear = " Keep doors and windows clear." if kind is TaskKind.NEXT_TO_CLEAR else ""
    return Task(task_id, kind, f"Move {_name(mover)} next to {_name(anchor)}." + clear,
                mover.object_id, anchor_id=anchor.object_id)


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
