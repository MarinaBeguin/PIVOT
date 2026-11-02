"""Tests of new furniture arriving exactly where the user puts it."""

import json
from typing import Any

import numpy as np

from pivot.agent.tools import Session, call_tool
from pivot.geometry.shape import VoxelShape
from pivot.scene.examples import box, office_scene, rectangle_floor, two_rooms_floor
from pivot.scene.graph import Scene, SceneObject


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


def with_door(scene: Scene, width: float) -> Scene:
    door = SceneObject(9, "door", box((0.05, 1.0, 1.0), (0.1, width, 2.0)), movable=False)
    return Scene.from_objects([*scene.objects.values(), door], scene.floor)


def test_a_new_sofa_goes_exactly_where_asked_and_says_which_doors_it_passes() -> None:
    session = Session(with_door(office_scene(), 0.8))
    sofa = call(session, "add_object", label="sofa", width_m=2.0, depth_m=0.9, height_m=0.8,
                x=3.5, y=1.0)
    assert sofa["added"] and (sofa["x"], sofa["y"], sofa["shift_m"]) == (3.5, 1.0, 0.0)
    assert sofa["doors"] == [{"id": 9, "width_m": 0.8, "fits_assembled": False}]
    stool = call(session, "add_object", label="stool", width_m=0.4, depth_m=0.4, height_m=0.5,
                 x=1.0, y=0.5)
    assert stool["doors"][0]["fits_assembled"]


def test_no_path_is_needed_to_put_new_furniture_in_a_room_behind_a_narrow_door() -> None:
    session = Session(Scene.from_objects([], two_rooms_floor(door_width=0.5)))
    sofa = call(session, "add_object", label="sofa", width_m=2.0, depth_m=0.9, height_m=0.8,
                x=6.0, y=2.0)
    assert sofa["added"] and sofa["x"] == 6.0
    # Once there, it moves like any other object: not through that door.
    back = call(session, "move_object", object_id=sofa["object_id"], x=2.0, y=2.0)
    assert not back["moved"]


def test_a_taken_spot_is_named_then_cleared_with_verified_moves() -> None:
    session = Session(office_scene())
    blocked = call(session, "add_object", label="armchair", width_m=0.8, depth_m=0.8,
                   height_m=0.9, x=1.5, y=1.6)
    assert not blocked["added"]
    assert blocked["in_the_way"] == [{"id": 3, "label": "office chair", "movable": True}]
    assert len(session.history) == 0
    placed = call(session, "add_object", label="armchair", width_m=0.8, depth_m=0.8,
                  height_m=0.9, x=1.5, y=1.6, make_room=True)
    assert placed["added"] and [s["object_id"] for s in placed["moved_aside"]] == [3]
    assert placed["moved_aside"][0]["status"] == "success"
    assert call(session, "undo")["undone"]
    assert np.allclose(session.scene[3].box.center, office_scene()[3].box.center)


def test_fixed_objects_and_walls_are_never_cleared() -> None:
    column = SceneObject(6, "column", box((3.0, 1.0, 1.0), (0.3, 0.3, 2.0)), movable=False)
    session = Session(Scene.from_objects([*office_scene().objects.values(), column],
                                         office_scene().floor))
    result = call(session, "add_object", label="armchair", width_m=0.8, depth_m=0.8,
                  height_m=0.9, x=3.0, y=1.0, make_room=True)
    assert not result["added"] and "cannot be moved" in result["reason"]
    wall = call(session, "add_object", label="armchair", width_m=0.8, depth_m=0.8,
                height_m=0.9, x=0.1, y=1.0)
    assert not wall["added"] and ("wall" in wall["reason"] or "floor" in wall["reason"])


def test_a_new_object_never_goes_under_a_table_but_may_stand_on_a_rug() -> None:
    # A carved table (top and four legs) and a rug, in a 4 x 3 m room.
    i, j, k = np.meshgrid(np.arange(-15, 15), np.arange(-10, 10), np.arange(-10, 10),
                          indexing="ij")
    cells = np.column_stack([i.ravel(), j.ravel(), k.ravel()])
    top = cells[:, 2] >= 7
    legs = (np.abs(cells[:, 0] + 0.5) >= 13) & (np.abs(cells[:, 1] + 0.5) >= 8)
    table = SceneObject(1, "table", box((2.0, 1.5, 0.38), (1.2, 0.8, 0.76)),
                        shape=VoxelShape(0.04, cells[top | legs]))
    rug = SceneObject(2, "rug", box((1.0, 1.0, 0.01), (1.2, 0.8, 0.02)))
    session = Session(Scene.from_objects([table, rug], rectangle_floor(4.0, 3.0)))
    under = call(session, "add_object", label="nightstand", width_m=0.4, depth_m=0.4,
                 height_m=0.5, x=2.0, y=1.5)
    assert not under["added"] and [o["id"] for o in under["in_the_way"]] == [1]
    on_rug = call(session, "add_object", label="nightstand", width_m=0.4, depth_m=0.4,
                  height_m=0.5, x=1.0, y=1.0)
    assert on_rug["added"] and on_rug["shift_m"] == 0.0
    # Placed wherever free, it does not end up under the table either.
    anywhere = call(session, "add_object", label="stool", width_m=0.4, depth_m=0.4,
                    height_m=0.5)
    assert anywhere["added"]
    assert abs(anywhere["x"] - 2.0) > 0.75 or abs(anywhere["y"] - 1.5) > 0.55
    # Once in the room, it can be pushed under the table like any small object.
    pushed = call(session, "move_object", object_id=on_rug["object_id"], x=2.0, y=1.5)
    assert pushed["moved"]
