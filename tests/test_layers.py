"""Tests of motion planning among objects with voxel shapes."""

import dataclasses
import json

from pivot.agent.tools import Session, call_tool
from pivot.geometry.collision import box_penetration
from pivot.planning.motion import MotionStatus, plan_motion
from pivot.planning.placement import check_placement
from pivot.scene.examples import box, legged_table, rectangle_floor
from pivot.scene.graph import Scene, SceneObject, carried_boxes, carried_objects


def scene(with_shape: bool = True) -> Scene:
    table = legged_table(1, (2.0, 1.5))
    if not with_shape:
        table = dataclasses.replace(table, shape=None)
    carton = SceneObject(2, "carton", box((0.6, 0.6, 0.2), (0.4, 0.4, 0.4)))
    return Scene.from_objects([table, carton], rectangle_floor(4.0, 3.0))


def test_a_carton_can_be_pushed_under_a_table() -> None:
    goal = scene()[2].box.moved(center_xy=scene()[1].box.center[:2])
    assert check_placement(scene(), 2, goal).feasible
    plan = plan_motion(scene(), 2, goal)
    assert plan.status is MotionStatus.SUCCESS, plan.message


def test_under_a_table_is_blocked_without_its_shape() -> None:
    goal = scene()[2].box.moved(center_xy=scene()[1].box.center[:2])
    assert not check_placement(scene(with_shape=False), 2, goal).feasible


def test_a_carton_cannot_stand_on_a_leg() -> None:
    leg_xy = scene()[1].box.center[:2] + [0.56, 0.36]
    goal = scene()[2].box.moved(center_xy=leg_xy)
    report = check_placement(scene(), 2, goal)
    assert not report.feasible
    assert report.conflicts[0].other_label == "table"


def desk_with_monitor() -> Scene:
    """A desk carrying a monitor along the top wall of a 6 x 3 m room, left of a wall shelf."""
    desk = legged_table(1, (1.0, 2.4))
    monitor = SceneObject(2, "monitor", box((1.0, 2.5, 0.98), (0.5, 0.2, 0.44)))
    # A wall shelf at 0.9-1.0 m: the desk fits under it, the monitor does not.
    shelf = SceneObject(3, "shelf", box((3.0, 2.7, 0.95), (1.2, 0.6, 0.1)), movable=False)
    return Scene.from_objects([desk, monitor, shelf], rectangle_floor(6.0, 3.0))


def test_carried_objects_are_checked_for_collisions() -> None:
    scene_ = desk_with_monitor()
    under_shelf = scene_[1].box.moved(center_xy=[3.0, 2.4])
    report = check_placement(scene_, 1, under_shelf)
    assert not report.feasible
    assert report.conflicts[0].carried_label == "monitor"
    assert report.conflicts[0].other_label == "shelf"
    assert "monitor (#2) it carries" in report.conflicts[0].describe()
    # Without the monitor, the desk alone fits under the shelf.
    alone = Scene.from_objects([scene_[1], scene_[3]], scene_.floor)
    assert check_placement(alone, 1, under_shelf).feasible


def test_planner_keeps_carried_objects_clear() -> None:
    scene_ = desk_with_monitor()
    # Along the wall, past the shelf: the monitor must not sweep under it.
    plan = plan_motion(scene_, 1, scene_[1].box.moved(center_xy=[5.0, 2.4]))
    assert plan.status is MotionStatus.SUCCESS, plan.message
    monitor_path = [carried_boxes(scene_, 1, b)[2] for b in plan.waypoints]
    shelf = scene_[3].box
    assert all(box_penetration(m, shelf) == 0.0 for m in monitor_path)
    assert min(b.center[1] for b in plan.waypoints) < 2.4 - 0.1  # It moved away from the wall.


def desk_under_narrow_shelf(monitor_y: float) -> Scene:
    """A desk (y 2.0-2.8) next to a 40 cm deep shelf overhanging its back by 20 cm."""
    desk = legged_table(1, (1.0, 2.4))
    monitor = SceneObject(2, "monitor", box((1.0, monitor_y, 0.98), (0.5, 0.15, 0.44)))
    shelf = SceneObject(3, "shelf", box((3.0, 2.8, 0.95), (1.2, 0.4, 0.1)), movable=False)
    return Scene.from_objects([desk, monitor, shelf], rectangle_floor(6.0, 3.0))


def test_shelf_overhang_only_blocks_what_is_under_it() -> None:
    under_shelf = [3.0, 2.4]
    front = desk_under_narrow_shelf(monitor_y=2.15)
    assert check_placement(front, 1, front[1].box.moved(center_xy=under_shelf)).feasible
    back = desk_under_narrow_shelf(monitor_y=2.65)
    report = check_placement(back, 1, back[1].box.moved(center_xy=under_shelf))
    assert [c.carried_label for c in report.conflicts] == ["monitor"]


def test_moving_the_monitor_forward_lets_the_desk_in() -> None:
    session = Session(desk_under_narrow_shelf(monitor_y=2.65))
    desk_goal = {"object_id": 1, "x": 3.0, "y": 2.4}
    assert not json.loads(call_tool(session, "move_object", json.dumps(desk_goal)))["moved"]
    monitor_forward = {"object_id": 2, "x": 1.0, "y": 2.15}
    assert json.loads(call_tool(session, "move_object", json.dumps(monitor_forward)))["moved"]
    assert carried_objects(session.scene, 1) == [2]  # Still on the desk.
    result = json.loads(call_tool(session, "move_object", json.dumps(desk_goal)))
    assert result["moved"] and result["carried_along"] == [2]
