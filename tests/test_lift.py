"""Tests of gravity and carried moves."""

import dataclasses
import json
from typing import Any

from pivot.agent.tools import Session, call_tool
from pivot.scene.examples import box, legged_table, rectangle_floor
from pivot.scene.graph import Scene, SceneObject


def office() -> Scene:
    """Two desks across a 4 x 3 m room, wall to wall; a monitor on one, a carton by the wall."""
    left = legged_table(1, (2.0, 0.9), size=(0.8, 1.8, 0.76))
    right = legged_table(2, (2.0, 2.4), size=(0.8, 1.2, 0.76))
    monitor = SceneObject(3, "monitor", box((2.0, 0.9, 0.98), (0.2, 0.5, 0.44)))
    # A low cabinet closes the gap between the desks: the floor is cut in two.
    cabinet = SceneObject(4, "cabinet", box((2.0, 1.9, 0.35), (0.8, 0.2, 0.7)), movable=False)
    # Too tall (75 cm) to pass under the desk tops (68 cm).
    carton = SceneObject(5, "carton", box((0.6, 2.4, 0.375), (0.4, 0.4, 0.75)))
    return Scene.from_objects([left, right, monitor, cabinet, carton], rectangle_floor(4.0, 3.0))


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


def test_an_object_taken_off_a_desk_rests_on_the_floor() -> None:
    session = Session(office())
    result = call(session, "move_object", object_id=3, x=3.5, y=0.5)
    assert result["moved"], result
    assert session.scene[3].box.z_min == 0.0
    heights = [w[3] for w in result["plan"]["waypoints"]]
    assert heights[0] > 0.7 and heights[-1] == 0.0  # Lifted off the desk, lowered at the end.


def test_an_object_can_be_put_on_another_desk() -> None:
    session = Session(office())
    result = call(session, "move_object", object_id=3, x=2.0, y=2.6, carry_height_m=0.8)
    assert result["moved"], result
    assert abs(session.scene[3].box.z_min - session.scene[2].box.z_max) < 1e-6


def test_a_light_object_is_carried_over_a_desk() -> None:
    session = Session(office())
    # Along the floor, the desks and the cabinet wall off the right half of the room.
    assert not call(session, "move_object", object_id=5, x=3.4, y=2.4)["moved"]
    result = call(session, "move_object", object_id=5, x=3.4, y=2.4, carry_height_m=0.85)
    assert result["moved"], result
    assert session.scene[5].box.z_min == 0.0


def test_objects_are_not_put_down_on_a_carton() -> None:
    session = Session(office())
    # Carried over the carton (a box as high as a desk), the monitor still goes down to the floor.
    result = call(session, "move_object", object_id=3, x=0.6, y=2.4, carry_height_m=0.9)
    assert not result["moved"] or session.scene[3].box.z_min == 0.0
    plan = call(session, "plan_motion", object_id=3, x=0.6, y=2.4, carry_height_m=0.9)
    assert plan["status"] != "success" or plan["waypoints"][-1][3] == 0.0


def test_a_carry_that_cannot_lift_falls_back_to_sliding() -> None:
    # A carton under the desk top cannot be lifted where it stands, but can slide out.
    carton = SceneObject(6, "carton", box((2.0, 2.4, 0.2), (0.3, 0.3, 0.4)))
    scene = Scene.from_objects([*office().objects.values(), carton], office().floor)
    session = Session(scene)
    result = call(session, "move_object", object_id=6, x=3.4, y=2.4, carry_height_m=0.9)
    assert result["moved"], result
    assert result["plan"]["message"].startswith("Slid along the floor instead of carried")


def test_nothing_rises_above_the_ceiling() -> None:
    lamp = SceneObject(6, "lamp", box((3.4, 0.4, 0.8), (0.4, 0.4, 1.6)))
    room = Scene.from_objects([*office().objects.values(), lamp], rectangle_floor(4.0, 3.0))
    # On the desk (76 cm), the lamp would reach 2.36 m.
    session = Session(dataclasses.replace(room, ceiling=2.3))
    result = call(session, "move_object", object_id=6, x=2.0, y=2.6, carry_height_m=0.8)
    assert not result["moved"]
    assert "ceiling" in json.dumps(result)
    assert session.scene[6].box.z_min == 0.0
    high = Session(dataclasses.replace(room, ceiling=2.6))
    assert call(high, "move_object", object_id=6, x=2.0, y=2.6, carry_height_m=0.8)["moved"]


def test_light_objects_are_carried_when_sliding_is_blocked() -> None:
    session = Session(office())
    # Sliding is walled off by the desks (see above): the carton is carried over them.
    result = call(session, "move_object", object_id=5, x=3.4, y=2.4, auto_carry=True)
    assert result["moved"], result
    assert result["plan"]["message"].startswith("Carried over the furniture")
    assert session.scene[5].box.z_min == 0.0
    # Where sliding works, the object slides.
    result = call(session, "move_object", object_id=5, x=3.4, y=1.2, auto_carry=True)
    assert result["moved"] and max(w[3] for w in result["plan"]["waypoints"]) == 0.0


def test_light_objects_go_onto_a_desk_and_tall_ones_do_not() -> None:
    session = Session(office())
    result = call(session, "move_object", object_id=3, x=2.0, y=2.6, auto_carry=True)
    assert result["moved"], result
    assert abs(session.scene[3].box.z_min - session.scene[2].box.z_max) < 1e-6
    # On the walled-off left half, like the carton: a floor lamp is too tall to carry over.
    lamp = SceneObject(6, "floor lamp", box((0.6, 0.5, 0.8), (0.4, 0.4, 1.6)))
    room = Session(Scene.from_objects([*office().objects.values(), lamp], office().floor))
    assert call(room, "move_object", object_id=6, x=0.6, y=1.4, auto_carry=True)["moved"]
    assert not call(room, "move_object", object_id=6, x=3.4, y=2.4, auto_carry=True)["moved"]
    assert not call(room, "move_object", object_id=6, x=2.0, y=2.6, auto_carry=True)["moved"]


def test_the_agent_is_told_what_can_be_carried() -> None:
    objects = {o["id"]: o for o in call(Session(office()), "list_objects")["objects"]}
    assert objects[3]["can_be_carried"] and objects[5]["can_be_carried"]
    assert not objects[1]["can_be_carried"]  # A desk takes two people.
