"""Tests of the passage check, the people needed and the moving instructions."""

import json

from pivot.agent.tools import Session, call_tool
from pivot.planning.access import check_access
from pivot.planning.handling import people_needed
from pivot.planning.instructions import instructions
from pivot.scene.examples import box, office_scene, rectangle_floor
from pivot.scene.graph import Scene, SceneObject
from pivot.viz.plan import instructions_page


def corridor(gap: float) -> Scene:
    """A 4 x 3 m room cut by a cabinet, with a passage of ``gap`` along the front wall."""
    door = SceneObject(1, "door", box((1.0, 0.05, 1.0), (0.9, 0.1, 2.0)), movable=False)
    length = 3.0 - gap
    cabinet = SceneObject(2, "cabinet", box((2.5, gap + length / 2, 0.5), (0.2, length, 1.0)))
    chair = SceneObject(3, "chair", box((3.5, 1.5, 0.45), (0.5, 0.5, 0.9)))
    return Scene.from_objects([door, cabinet, chair], rectangle_floor(4.0, 3.0))


def test_a_narrow_passage_lets_a_person_through_but_not_a_wheelchair() -> None:
    walking = check_access(corridor(0.7), 0.6)
    wheelchair = check_access(corridor(0.7), 0.9)
    assert walking is not None and wheelchair is not None
    assert walking.from_doors and walking.to_dict()["accessible"]
    assert wheelchair.unreachable == [3]
    assert wheelchair.reachable_share < walking.reachable_share


def test_a_door_with_furniture_in_front_is_blocked() -> None:
    scene = corridor(1.0)
    crate = SceneObject(4, "crate", box((1.0, 0.5, 0.4), (1.4, 0.9, 0.8)))
    report = check_access(Scene.from_objects([*scene.objects.values(), crate], scene.floor), 0.6)
    assert report is not None
    assert report.blocked_doors == [1] and not report.from_doors


def test_the_agent_tool_reports_passages() -> None:
    result = json.loads(call_tool(Session(corridor(0.7)), "check_access", '{"width_m": 0.9}'))
    assert result["unreachable_objects"] == [{"id": 3, "label": "chair"}]
    assert not result["accessible"]


def test_bulky_or_heavy_objects_take_two_people() -> None:
    def obj(label: str, size: tuple[float, float, float]) -> SceneObject:
        return SceneObject(1, label, box((0.0, 0.0, size[2] / 2), size))

    assert people_needed(obj("sofa", (0.8, 0.8, 0.8))) == 2
    assert people_needed(obj("lamp", (1.0, 0.3, 2.0))) == 1
    assert people_needed(obj("table", (1.5, 0.9, 0.75))) == 2
    assert people_needed(obj("box", (0.4, 0.4, 0.4))) == 1
    assert people_needed(obj("crate", (0.5, 0.5, 0.5))) == 1


def test_moves_become_steps_a_person_can_follow() -> None:
    session = Session(office_scene())
    initial = session.scene
    assert json.loads(call_tool(session, "move_object", json.dumps(
        {"object_id": 3, "x": 3.0, "y": 1.0, "yaw_deg": 90})))["moved"]
    steps = instructions(initial, session.motions)
    assert len(steps) == 1
    step = steps[0]
    assert step.label == "office chair" and not step.carried and step.people == 1
    assert abs(abs(step.turn_deg) - 90) < 1
    text = step.text()
    assert text.startswith("Slide the office chair (#3) ")
    assert "turning it 90 degrees" in text
    page = instructions_page(session.scene, [step], "office")
    assert "<svg" in page and "Slide the office chair" in page
    assert "wheelchair" in page
