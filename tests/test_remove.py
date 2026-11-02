"""Tests of taking objects out of the room, and of the points that go with objects."""

import json
from typing import Any

import numpy as np

from pivot.agent.tools import Session, call_tool
from pivot.lifting.completion import complete_rugs
from pivot.planning.instructions import instructions
from pivot.scene.examples import box, office_scene, rectangle_floor
from pivot.scene.graph import Scene, SceneObject
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


def test_a_rug_half_under_a_sofa_stays_whole_when_the_sofa_goes() -> None:
    door = SceneObject(1, "door", box((0.5, 0.05, 1.0), (0.9, 0.1, 2.0)), movable=False)
    # The rug really spans x in [1, 3]; the sofa hides x > 2, so only the half before is seen.
    seen_rug = SceneObject(2, "rug", box((1.5, 2.0, 0.01), (1.0, 1.4, 0.02)))
    sofa = SceneObject(3, "sofa", box((2.6, 2.0, 0.4), (1.2, 1.8, 0.8)))
    scene = Scene.from_objects([door, seen_rug, sofa], rectangle_floor(5.0, 4.0))
    points = np.array([[1.5, 2.0, 0.02], [1.2, 1.5, 0.02], [2.6, 2.0, 0.8]])
    colors = np.array([[200, 30, 30], [200, 30, 30], [90, 90, 90]], dtype=np.uint8)
    completed, cloud, cloud_colors = complete_rugs(scene, points, colors)
    rug = completed[2].box
    # It goes on under the sofa up to the sofa's far side, not beyond; it keeps its width.
    assert np.allclose([rug.center[0] - rug.size[0] / 2, rug.center[0] + rug.size[0] / 2],
                       [1.0, 3.2], atol=0.03)
    assert np.isclose(rug.size[1], 1.4)
    hidden = cloud[len(points):]
    assert len(hidden) and np.all(hidden[:, 0] > 2.0) and np.all(cloud_colors[len(points):]
                                                                   == [200, 30, 30])
    again, cloud_again, _ = complete_rugs(completed, cloud, cloud_colors)
    assert again[2].box == rug and len(cloud_again) == len(cloud)
    session = Session(completed)
    assert call(session, "remove_object", object_id=3)["removed"]
    assert session.scene[2].box == rug
