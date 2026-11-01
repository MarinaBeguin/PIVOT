"""Tests of taking objects out of the room, and of the points that go with objects."""

import json
from typing import Any

import numpy as np

from pivot.agent.tools import Session, call_tool
from pivot.planning.instructions import instructions
from pivot.scene.examples import office_scene
from pivot.viz.ownership import point_owners
from pivot.viz.plan import instructions_page


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


def test_a_desk_goes_out_with_its_lamp_and_undo_brings_both_back() -> None:
    session = Session(office_scene())
    result = call(session, "remove_object", object_id=1)
    assert result["removed"] and result["removed_with"] == [{"id": 2, "label": "lamp"}]
    assert 1 not in session.scene.objects and 2 not in session.scene.objects
    assert result["floor_freed_m2"] == 0.98
    assert call(session, "undo")["undone"]
    assert {1, 2} <= session.scene.objects.keys()


def test_fixed_objects_stay_and_unknown_ones_are_reported() -> None:
    session = Session(office_scene())
    window = call(session, "remove_object", object_id=5)
    assert not window["removed"] and "fixed" in window["reason"]
    assert "error" in call(session, "remove_object", object_id=999)
    assert len(session.scene) == len(office_scene())


def test_the_moving_plan_starts_with_what_goes_out() -> None:
    scene = office_scene()
    session = Session(scene)
    call(session, "remove_object", object_id=3)
    page = instructions_page(session.scene, instructions(scene, session.motions), "office",
                             removed=[("office chair", 3)])
    assert "First take out of the room: the office chair (#3)." in page


def test_points_belong_to_the_object_they_lie_in_but_the_floor_stays() -> None:
    scene = office_scene()
    rng = np.random.default_rng(0)
    chair = rng.uniform([1.25, 1.45, 0.1], [1.75, 1.95, 0.95], (200, 3))  # Office chair #3.
    floor_under = np.column_stack([rng.uniform(1.3, 1.7, 50), rng.uniform(1.5, 1.9, 50),
                                   np.full(50, 0.01)])
    lamp = rng.uniform([1.15, 2.55, 0.8], [1.25, 2.65, 1.1], (50, 3))  # On the desk.
    window = rng.uniform([2.3, 3.9, 1.1], [3.3, 4.0, 2.1], (50, 3))  # Fixed.
    owners = point_owners(scene, np.vstack([chair, floor_under, lamp, window]))
    assert (owners[:200] == 3).all() and (owners[200:250] == -1).all()
    assert (owners[250:300] == 2).all()  # The smaller object wins over the desk.
    assert (owners[300:] == -1).all()
