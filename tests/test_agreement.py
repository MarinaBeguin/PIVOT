"""Tests of checking objects against what the camera saw, and of refitting their boxes."""

import numpy as np
import pytest

from pivot.lifting.agreement import agree_with_camera
from pivot.lifting.openings import Walls
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject


def block(lo: tuple[float, float, float], hi: tuple[float, float, float], n: int,
          seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(lo, hi, (n, 3))


def test_a_box_set_beside_its_table_moves_onto_it() -> None:
    top = block((1.0, 1.0, 0.7), (2.4, 1.6, 0.76), 4000)  # The table seen by the camera.
    aisle = block((1.0, 1.6, 0.1), (2.4, 2.4, 0.9), 4000, 1)  # Chairs in the aisle beside it.
    # Its box is set half on the aisle.
    misplaced = SceneObject(1, "table", box((1.7, 1.6, 0.38), (1.4, 0.6, 0.76)))
    (moved,), scores = agree_with_camera([misplaced], {1: top}, np.vstack([top, aisle]))
    assert moved.box.center[1] == pytest.approx(1.3, abs=0.06)
    _, before = agree_with_camera([misplaced], {1: top}, np.vstack([top, aisle]), refit=False)
    assert scores[1] > before[1] + 0.2


def test_a_refit_never_cuts_into_the_object() -> None:
    # A sofa whose box holds the sofa and some of the wall behind: trimming the wall is
    # fine, cutting into the sofa is not.
    sofa = block((0.0, 0.0, 0.0), (2.0, 0.9, 0.8), 6000)
    wall = block((0.0, 0.9, 0.0), (2.0, 1.0, 0.8), 1500, 2)
    obj = SceneObject(1, "sofa", box((1.0, 0.5, 0.4), (2.0, 1.0, 0.8)))
    (refit,), _ = agree_with_camera([obj], {1: sofa}, np.vstack([sofa, wall]))
    local = refit.box.to_local(sofa)
    inside = np.all(np.abs(local) <= 0.5 * refit.box.size + 1e-9, axis=1)
    assert inside.mean() >= 0.97


def test_doors_are_not_scored_and_fixed_objects_not_moved() -> None:
    seen = block((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 2000)
    door = SceneObject(1, "door", box((0.5, 0.0, 1.0), (0.9, 0.1, 2.0)))
    board = SceneObject(2, "board", box((0.8, 0.5, 0.5), (1.0, 1.0, 1.0)), movable=False)
    result, scores = agree_with_camera([door, board], {1: seen, 2: seen}, seen)
    assert 1 not in scores and 2 in scores
    assert result[1] is board


def walls_of(width: float, depth: float) -> Walls:
    return Walls(0.0, np.array([0.0, 0.0]), np.array([width, depth]))


def test_a_chair_past_a_wall_is_brought_back_in() -> None:
    chair = block((3.6, 1.0, 0.0), (4.1, 1.5, 0.9), 2000)  # Its points reach 10 cm past x = 4.
    obj = SceneObject(1, "chair", box((3.85, 1.25, 0.45), (0.5, 0.5, 0.9)))
    (inside,), _ = agree_with_camera([obj], {1: chair}, chair, walls=walls_of(4.0, 3.0))
    assert inside.box.center[0] + 0.5 * inside.box.size[0] == pytest.approx(4.0, abs=0.01)


def test_a_round_lamp_is_set_square_to_the_walls_and_a_turned_chair_is_not() -> None:
    rng = np.random.default_rng(4)
    angle, radius = rng.uniform(0, 2 * np.pi, 3000), 0.2 * np.sqrt(rng.uniform(0, 1, 3000))
    lamp = np.column_stack([1 + radius * np.cos(angle), 1 + radius * np.sin(angle),
                            rng.uniform(0, 1.5, 3000)])
    turned = SceneObject(1, "lamp", box((1.0, 1.0, 0.75), (0.4, 0.4, 1.5), np.pi / 4))
    # A chair truly turned by 30 degrees: its square box would add empty corners.
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    seat = block((-0.3, -0.25, 0.0), (0.3, 0.25, 0.9), 3000, 5)
    chair_points = np.column_stack([seat[:, :2] @ np.array([[c, s], [-s, c]]) + [3.0, 1.0],
                                    seat[:, 2]])
    chair = SceneObject(2, "chair", box((3.0, 1.0, 0.45), (0.6, 0.5, 0.9), np.pi / 6))
    result, _ = agree_with_camera([turned, chair], {1: lamp, 2: chair_points},
                                  np.vstack([lamp, chair_points]), room_yaw=0.0)
    assert abs(np.sin(2 * result[0].box.yaw)) < 1e-6  # Square to the walls.
    assert result[1].box.yaw == pytest.approx(np.pi / 6)
