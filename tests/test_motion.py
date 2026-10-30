"""Tests for configuration-space motion planning."""

import math

import numpy as np
import pytest

from conftest import box
from pivot.scene.examples import sofa_scene
from pivot.geometry.floor import FloorMap
from pivot.planning.cspace import ConfigurationSpace, footprint_kernel
from pivot.planning.motion import (
    MotionOptions, MotionStatus, _aliasing, plan_motion, straight_motion,
)
from pivot.planning.placement import Conflict, ConflictKind, PlacementReport, check_placement
from pivot.scene.graph import Scene, SceneObject

SOFA = (2.0, 0.9, 0.75)


def assert_path_is_free(scene: Scene, object_id: int, waypoints: list) -> None:
    for pose in waypoints:
        assert check_placement(scene, object_id, pose).feasible


def test_footprint_kernel_is_rotated() -> None:
    kernel = footprint_kernel(np.array([0.5, 0.1]), math.pi / 2, resolution=0.05)
    rows, cols = np.nonzero(kernel)
    assert np.ptp(rows) > np.ptp(cols)
    assert kernel[kernel.shape[0] // 2, kernel.shape[1] // 2]


def test_cspace_blocks_walls_for_a_wide_object() -> None:
    cspace = ConfigurationSpace.build(sofa_scene(0.8), 1)
    _, door_col = cspace.nearest_cell(np.array([4.05, 2.0]))
    door_row, _ = cspace.nearest_cell(np.array([4.05, 2.0]))
    assert not cspace.free[:, door_row, door_col].any()


def test_sofa_cannot_pass_narrow_door() -> None:
    scene = sofa_scene(0.8)
    goal = scene[1].box.moved(np.array([6.0, 2.0]))
    # The destination itself is fine: only the motion is impossible.
    assert check_placement(scene, 1, goal).feasible
    plan = plan_motion(scene, 1, goal)
    assert plan.status is MotionStatus.NO_PATH
    assert not plan.waypoints


def test_sofa_turns_to_pass_wide_door() -> None:
    scene = sofa_scene(1.1)
    goal = scene[1].box.moved(np.array([6.0, 2.0]))
    plan = plan_motion(scene, 1, goal)
    assert plan.status is MotionStatus.SUCCESS, plan.message
    np.testing.assert_allclose(plan.waypoints[0].center, scene[1].box.center)
    np.testing.assert_allclose(plan.waypoints[-1].center, goal.center)
    # Lengthwise across the door it is 2 m wide: it must turn. A diagonal
    # crossing would fit (0.9 m / cos(22.5 deg) < 1.1 m) but grazes the frame;
    # the planner prefers turning square to the door and back.
    assert plan.rotation == pytest.approx(math.pi, abs=1e-6)
    assert_path_is_free(scene, 1, plan.waypoints)
    # The door frame is the bottleneck: about 10 cm on each side.
    assert plan.min_clearance is not None and 0.05 <= plan.min_clearance < 0.12
    assert plan.to_dict()["status"] == "success"
    # Shortcutting leaves a handful of straight segments, not a staircase.
    assert len(plan.waypoints) <= 10


def test_wall_clearance_closes_tight_door() -> None:
    scene = sofa_scene(1.1)
    goal = scene[1].box.moved(np.array([6.0, 2.0]))
    # 0.97 m of sofa cross-section in a 1.1 m door leaves under 10 cm per side.
    plan = plan_motion(scene, 1, goal, MotionOptions(clearance=0.15))
    assert plan.status is MotionStatus.NO_PATH


def test_detours_around_obstacle() -> None:
    xs, ys = np.meshgrid(np.arange(0.0, 5.0, 0.02), np.arange(0.0, 3.0, 0.02))
    floor = FloorMap.from_points(np.column_stack([xs.ravel(), ys.ravel()]), close_radius=0)
    scene = Scene.from_objects([
        SceneObject(1, "chair", box((0.6, 1.5, 0.5), (0.5, 0.5, 1.0))),
        # 70 cm passages on both sides of the table.
        SceneObject(2, "table", box((2.5, 1.5, 0.375), (0.8, 1.6, 0.75))),
    ], floor)
    goal = scene[1].box.moved(np.array([4.4, 1.5]))
    plan = plan_motion(scene, 1, goal, MotionOptions(clearance=0.05))
    assert plan.status is MotionStatus.SUCCESS, plan.message
    assert plan.length > 3.8 + 0.2
    assert plan.min_clearance is not None and plan.min_clearance >= 0.05 - 1e-9


@pytest.mark.parametrize(("height", "expected"), [
    (0.45, MotionStatus.SUCCESS),
    (1.0, MotionStatus.NO_PATH),
])
def test_low_objects_slide_under_overhangs(height: float, expected: MotionStatus) -> None:
    xs, ys = np.meshgrid(np.arange(0.0, 5.0, 0.02), np.arange(0.0, 2.0, 0.02))
    floor = FloorMap.from_points(np.column_stack([xs.ravel(), ys.ravel()]), close_radius=0)
    scene = Scene.from_objects([
        SceneObject(1, "stool", box((0.5, 1.0, height / 2), (0.4, 0.4, height))),
        # A counter overhang spanning the whole corridor, 60 cm above the floor.
        SceneObject(2, "counter", box((2.5, 1.0, 0.8), (0.6, 2.0, 0.4)), movable=False),
    ], floor)
    plan = plan_motion(scene, 1, scene[1].box.moved(np.array([4.5, 1.0])))
    assert plan.status is expected, plan.message


def test_invalid_goal_is_explained() -> None:
    scene = sofa_scene(1.1)
    plan = plan_motion(scene, 1, scene[1].box.moved(np.array([4.05, 0.5])))
    assert plan.status is MotionStatus.INVALID_GOAL
    assert "walkable floor" in plan.message


def test_budget_exhaustion_is_not_reported_as_impossible() -> None:
    scene = sofa_scene(1.1)
    goal = scene[1].box.moved(np.array([6.0, 2.0]))
    plan = plan_motion(scene, 1, goal, MotionOptions(max_expansions=10))
    assert plan.status is MotionStatus.SEARCH_LIMIT_REACHED


def diagonal_corridor_scene() -> Scene:
    """Two 2 x 2 m rooms joined by a 0.6 m wide corridor running at 45 degrees."""
    xs, ys = np.meshgrid(np.arange(0.0, 6.0, 0.02), np.arange(0.0, 6.0, 0.02))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    in_rooms = np.all(points < 2.0, axis=1) | np.all(points > 4.0, axis=1)
    along = (points[:, 0] + points[:, 1]) / math.sqrt(2.0)
    across = (points[:, 1] - points[:, 0]) / math.sqrt(2.0)
    in_corridor = (np.abs(across) < 0.3) & (along > 1.5) & (along < 7.0)
    floor = FloorMap.from_points(points[in_rooms | in_corridor], resolution=0.05, close_radius=0)
    plank = SceneObject(1, "plank", box((1.0, 1.0, 0.1), (1.2, 0.3, 0.2)))
    return Scene.from_objects([plank], floor)


def test_refined_angles_find_intermediate_passage() -> None:
    scene = diagonal_corridor_scene()
    goal = scene[1].box.moved(np.array([5.0, 5.0]))
    # With quarter turns only, the plank never aligns with the corridor.
    coarse = MotionOptions(yaw_bins=2, refine_yaw_bins=2, comfort=0.0)
    assert plan_motion(scene, 1, goal, coarse).status is MotionStatus.NO_PATH
    refined = plan_motion(scene, 1, goal, MotionOptions(yaw_bins=2, refine_yaw_bins=4,
                                                         comfort=0.0))
    assert refined.status is MotionStatus.SUCCESS, refined.message
    assert_path_is_free(scene, 1, refined.waypoints)


def test_effort_counts_translation_and_corner_arcs() -> None:
    scene = sofa_scene(1.1)
    plan = plan_motion(scene, 1, scene[1].box.moved(np.array([6.0, 2.0])))
    half_diagonal = 0.5 * math.hypot(*SOFA[:2])
    assert plan.effort == pytest.approx(plan.length + half_diagonal * plan.rotation)
    # Grazing the frame diagonally saves effort; keeping a comfort margin costs some.
    tight = plan_motion(scene, 1, scene[1].box.moved(np.array([6.0, 2.0])),
                        MotionOptions(comfort=0.0))
    assert tight.effort < plan.effort


def test_straight_motion_slides_only_when_nothing_is_in_the_way() -> None:
    xs, ys = np.meshgrid(np.arange(0.0, 5.0, 0.02), np.arange(0.0, 3.0, 0.02))
    floor = FloorMap.from_points(np.column_stack([xs.ravel(), ys.ravel()]), close_radius=0)
    scene = Scene.from_objects([
        SceneObject(1, "chair", box((0.6, 1.5, 0.5), (0.5, 0.5, 1.0))),
        SceneObject(2, "table", box((2.5, 1.5, 0.375), (0.8, 1.6, 0.75))),
    ], floor)
    aside = straight_motion(scene, 1, scene[1].box.moved(np.array([0.6, 0.5])))
    assert aside is not None and aside.status is MotionStatus.SUCCESS
    assert straight_motion(scene, 1, scene[1].box.moved(np.array([4.4, 1.5]))) is None


def test_only_a_shallow_collision_can_be_voxel_aliasing() -> None:
    def report(*conflicts: Conflict) -> PlacementReport:
        return PlacementReport(1, box((0, 0, 0), (1, 1, 1)), list(conflicts))

    assert _aliasing(report(Conflict(ConflictKind.COLLISION, 0.04, 2)))
    assert not _aliasing(report(Conflict(ConflictKind.COLLISION, 0.12, 2)))
    assert not _aliasing(report(Conflict(ConflictKind.WALL_CLEARANCE, 0.01)))
