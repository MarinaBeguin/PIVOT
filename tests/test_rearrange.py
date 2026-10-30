"""Tests of moving blocking objects aside."""

import json
from typing import Any

import numpy as np

from pivot.agent.tools import Session, call_tool
from pivot.planning.motion import MotionStatus, plan_motion
from pivot.planning.rearrange import plan_rearrangement
from pivot.scene.examples import box, rectangle_floor
from pivot.scene.graph import Scene, SceneObject


def blocked_passage() -> Scene:
    """A fixed cabinet leaves a 1.1 m passage at the top; a chair stands in it."""
    cabinet = SceneObject(1, "cabinet", box((3.0, 0.95, 0.5), (0.4, 1.9, 1.0)), movable=False)
    chair = SceneObject(2, "chair", box((3.0, 2.45, 0.45), (0.5, 0.5, 0.9)))
    carton = SceneObject(3, "carton", box((1.0, 1.0, 0.3), (0.6, 0.6, 0.6)))
    return Scene.from_objects([cabinet, chair, carton], rectangle_floor(6.0, 3.0))


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


def test_chair_in_the_passage_is_moved_aside() -> None:
    scene = blocked_passage()
    goal = scene[3].box.moved(center_xy=np.array([5.0, 1.0]))
    assert plan_motion(scene, 3, goal).status is MotionStatus.NO_PATH
    plan = plan_rearrangement(scene, 3, goal)
    assert plan.success, plan.message
    assert plan.blockers == [2]
    # Aside, the carton, then the chair back as close as possible to its place.
    assert [step.object_id for step in plan.steps] == [2, 3, 2]
    assert "chair (#2)" in plan.message and "put back" in plan.message
    back = plan.steps[-1].waypoints[-1]
    assert np.linalg.norm(back.center[:2] - scene[2].box.center[:2]) < 1.0


def test_blockers_can_stay_aside() -> None:
    scene = blocked_passage()
    goal = scene[3].box.moved(center_xy=np.array([5.0, 1.0]))
    plan = plan_rearrangement(scene, 3, goal, put_back=False)
    assert [step.object_id for step in plan.steps] == [2, 3]


def test_fixed_obstacles_are_never_moved() -> None:
    scene = blocked_passage()
    walled = Scene.from_objects([*scene.objects.values(), SceneObject(
        4, "wall unit", box((3.0, 2.45, 0.5), (0.4, 1.1, 1.0)), movable=False)], scene.floor)
    goal = walled[3].box.moved(center_xy=np.array([5.0, 1.0]))
    plan = plan_rearrangement(walled, 3, goal)
    assert not plan.success and "Even with every movable object out of the way" in plan.message


def test_make_room_and_move_runs_all_steps_and_one_undo_reverts_them() -> None:
    session = Session(blocked_passage())
    assert not call(session, "move_object", object_id=3, x=5.0, y=1.0)["moved"]
    preview = call(session, "make_room", object_id=3, x=5.0, y=1.0)
    assert preview["success"] and session.scene[3].box.center[0] == 1.0  # Nothing moved.
    result = call(session, "make_room_and_move", object_id=3, x=5.0, y=1.0)
    assert result["moved"]
    assert [step["object_id"] for step in result["steps"]] == [2, 3, 2]
    assert session.scene[3].box.center[:2].tolist() == [5.0, 1.0]
    assert len(session.motions) == 3
    assert call(session, "undo")["undone"]
    assert not session.motions  # The three steps were one move.
    assert session.scene[2].box.center[:2].tolist() == [3.0, 2.45]
    assert session.scene[3].box.center[:2].tolist() == [1.0, 1.0]
