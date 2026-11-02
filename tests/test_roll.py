"""Tests of moving a rug rolled up when it does not pass flat."""

import json
from typing import Any

import numpy as np
import pytest

from pivot.agent.tools import Session, call_tool
from pivot.planning.instructions import instructions
from pivot.planning.roll import ROLL_DIAMETER, rolled_box
from pivot.scene.examples import box, two_rooms_floor
from pivot.scene.graph import Scene, SceneObject


def rug_scene() -> Scene:
    """A 1.4 x 2.0 m rug in the left of two rooms joined by an 80 cm door."""
    rug = SceneObject(1, "rug", box((2.0, 2.0, 0.01), (1.4, 2.0, 0.02)))
    return Scene.from_objects([rug], floor=two_rooms_floor(0.8))


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


def test_a_roll_is_as_long_as_the_short_side() -> None:
    roll = rolled_box(box((0.0, 0.0, 0.01), (1.4, 2.0, 0.02), yaw=0.3))
    assert np.allclose(roll.size, [1.4, ROLL_DIAMETER, ROLL_DIAMETER])
    assert roll.z_min == pytest.approx(0.0) and roll.yaw == pytest.approx(0.3)


def test_a_rug_wider_than_the_door_goes_through_rolled_up() -> None:
    session = Session(rug_scene())
    result = call(session, "move_object", object_id=1, x=6.0, y=2.0)
    assert result["moved"] and result["plan"]["rolled_up"]
    # It lies flat, whole, at its place.
    rug = session.scene[1].box
    assert np.allclose(rug.size, [1.4, 2.0, 0.02]) and np.allclose(rug.center[:2], [6.0, 2.0])
    step = instructions(rug_scene(), session.motions)[0]
    assert step.text().startswith("Roll up the rug (#1)") and "Unroll it there." in step.text()


def test_a_rug_that_passes_flat_is_not_rolled() -> None:
    session = Session(rug_scene())
    result = call(session, "move_object", object_id=1, x=2.0, y=2.2)
    assert result["moved"] and "rolled_up" not in result["plan"]
