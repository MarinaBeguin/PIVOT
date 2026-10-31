"""Tests of rooms: splitting the floor at doorways, naming rooms, and moving between them."""

import json
from typing import Any

import numpy as np
import pytest

from pivot.agent.tools import Session, call_tool
from pivot.planning.instructions import instructions
from pivot.scene.examples import apartment_scene, box, office_scene, sofa_scene
from pivot.scene.graph import Scene, SceneObject
from pivot.scene.rooms import find_rooms
from pivot.viz.web import scene_payload


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


@pytest.mark.parametrize("door", [0.8, 0.9, 1.2])
def test_a_doorway_splits_two_rooms_and_its_width_is_measured(door: float) -> None:
    rooms = find_rooms(sofa_scene(door))
    assert rooms is not None and len(rooms.rooms) == 2
    assert [r.area for r in rooms.rooms] == pytest.approx([16.0, 16.0], abs=0.3)
    assert len(rooms.passages) == 1
    passage = rooms.passages[0]
    assert passage.rooms == (1, 2)
    assert passage.width == pytest.approx(door, abs=0.05)
    assert passage.center == pytest.approx([4.05, 2.0], abs=0.1)


def test_open_plans_and_single_rooms_stay_whole() -> None:
    wide = find_rooms(sofa_scene(1.6))  # Wider than any door: one open space.
    office = find_rooms(office_scene())
    assert wide is not None and len(wide.rooms) == 1 and not wide.passages
    assert office is not None and [r.name for r in office.rooms] == ["office"]


def test_rooms_are_named_after_their_objects() -> None:
    rooms = find_rooms(apartment_scene())
    assert rooms is not None
    assert [r.name for r in rooms.rooms] == ["living room", "bedroom"]
    assert rooms.object_rooms[1] == 1 and rooms.object_rooms[5] == 2
    assert rooms.passages[0].door_id == 6
    assert rooms.find("Bedroom") is rooms.rooms[1] and rooms.find("garage") is None
    two_beds = Scene.from_objects(
        [SceneObject(1, "bed", box((2.0, 2.0, 0.25), (2.0, 1.4, 0.5))),
         SceneObject(2, "bed", box((6.0, 2.0, 0.25), (2.0, 1.4, 0.5)))], sofa_scene(0.9).floor)
    named = find_rooms(two_beds)
    assert named is not None and [r.name for r in named.rooms] == ["bedroom 1", "bedroom 2"]


def test_the_agent_sees_rooms_and_doorways() -> None:
    result = call(Session(apartment_scene()), "list_objects")
    assert [r["name"] for r in result["rooms"]] == ["living room", "bedroom"]
    assert result["rooms"][0]["doorways"] == [{"to": "bedroom", "width_m": 0.9, "door_id": 6}]
    objects = {o["id"]: o for o in result["objects"]}
    assert objects[2]["room"] == "living room" and objects[5]["room"] == "bedroom"
    assert "rooms" not in call(Session(office_scene()), "list_objects")


def test_an_armchair_goes_to_the_bedroom_through_the_door() -> None:
    session = Session(apartment_scene())
    checked = call(session, "plan_to_room", object_id=2, room="bedroom")
    assert checked["found"] and not checked["moved"]
    assert session.scene[2].box.center[0] < 4.0  # Checking moves nothing.
    moved = call(session, "move_to_room", object_id=2, room="bedroom")
    assert moved["moved"] and moved["rooms_crossed"] == ["living room", "bedroom"]
    assert len(moved["doorways"]) == 1
    doorway = moved["doorways"][0]
    assert doorway["door_id"] == 6 and doorway["width_m"] == 0.9 and doorway["margin_m"] >= 0.0
    assert session.scene[2].box.center[0] > 4.1  # Now in the bedroom.
    assert not moved.get("warnings")  # Not left in front of the door.
    rooms = find_rooms(session.scene)
    assert rooms is not None and rooms.object_rooms[2] == 2
    steps = instructions(apartment_scene(), session.motions)
    assert len(steps) == 1
    step = steps[0]
    assert "from the living room to the bedroom (door #6" in step.text()


def test_a_sofa_wider_than_the_door_is_refused_at_once() -> None:
    result = call(Session(apartment_scene(0.9)), "plan_to_room", object_id=1, room="bedroom")
    assert not result["found"]
    assert "90 cm wide" in result["too_wide"] and "door #6" in result["too_wide"]


def test_unknown_rooms_and_single_rooms_are_reported() -> None:
    assert call(Session(apartment_scene()), "move_to_room", object_id=2,
                room="garage")["rooms"] == ["living room", "bedroom"]
    assert "single room" in call(Session(office_scene()), "plan_to_room", object_id=3,
                                 room="office")["error"]


def test_the_viewer_colors_the_floor_by_room() -> None:
    scene = apartment_scene()
    points, colors = np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8)
    payload = scene_payload(scene, points, colors, "apartment")
    assert [r["name"] for r in payload["rooms"]] == ["living room", "bedroom"]
    floor = payload["floor"]
    assert [len(r) for r in floor["rooms"]] == [len(c) for c in floor["cells"]]
    assert scene_payload(office_scene(), points, colors, "office")["rooms"] == []
