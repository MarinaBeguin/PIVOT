"""Tests of clearing a way for a wheelchair, moving as few objects as possible."""

import json
import math
from typing import Any

import numpy as np

from pivot.agent.tools import Session, call_tool
from pivot.planning.motion import MotionOptions
from pivot.planning.rearrange import park_aside
from pivot.planning.way import plan_way
from pivot.scene.examples import box, rectangle_floor
from pivot.scene.graph import Scene, SceneObject

DOOR = SceneObject(1, "door", box((0.05, 1.5, 1.0), (0.9, 0.1, 2.0), yaw=math.pi / 2),
                   movable=False)
# The table to reach, against the right wall of a 6 x 3 m room.
TABLE = SceneObject(2, "table", box((5.5, 1.5, 0.375), (0.8, 1.6, 0.75)))


def chair(object_id: int, x: float, y: float) -> SceneObject:
    return SceneObject(object_id, "chair", box((x, y, 0.45), (0.5, 0.5, 0.9)))


def room(*objects: SceneObject) -> Scene:
    return Scene.from_objects([DOOR, TABLE, *objects], floor=rectangle_floor(6.0, 3.0))


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


def test_the_way_goes_around_a_chair_rather_than_moving_it() -> None:
    way = plan_way(room(chair(3, 3.0, 1.5)), 2)
    assert way.success and not way.moved and way.door_id == 1


def test_a_row_of_chairs_is_opened_by_moving_one() -> None:
    row = [chair(3 + i, 3.0, y) for i, y in enumerate((0.3, 1.1, 1.9, 2.7))]
    session = Session(room(*row))
    result = call(session, "make_way", object_id=2)
    assert result["success"] and result["moved"] and len(result["moved_aside"]) == 1
    assert len(session.history) == 1
    # One undo puts everything back.
    assert call(session, "undo")["undone"]
    assert session.scene[3].box.center[1] == row[0].box.center[1]


def test_a_table_is_pushed_back_with_its_chairs() -> None:
    desk = SceneObject(3, "table", box((3.0, 1.5, 0.375), (0.8, 1.8, 0.75)))
    # A chair on each side: the table cannot slide alone.
    scene = room(desk, chair(4, 3.66, 1.5), chair(5, 2.34, 1.5))
    # The way to clear: a band across the room, through the table.
    band = [box((3.0, y, 0.5), (1.0, 0.5, 1.0)) for y in np.arange(0.25, 3.0, 0.5)]
    motions = park_aside(scene, 3, band, MotionOptions())
    assert motions is not None and {m.object_id for m in motions} == {3, 4, 5}
    shifts = [m.waypoints[-1].center[:2] - scene[m.object_id].box.center[:2] for m in motions]
    assert np.allclose(shifts[0], shifts[1]) and np.allclose(shifts[0], shifts[2])
