"""Tests of setting doors and windows in the walls."""

import math

import numpy as np
import pytest

from pivot.lifting.openings import Walls, clean_openings, snap_opening
from pivot.scene.examples import box, two_rooms_floor
from pivot.scene.graph import SceneObject

# A 4 x 3 m room, axis-aligned.
WALLS = Walls(0.0, np.array([0.0, 0.0]), np.array([4.0, 3.0]))


def window_points(x_range: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Points of a window in the wall y = 3, plus the street seen through it, 2 m behind."""
    glass = rng.uniform([x_range[0], 2.95, 1.0], [x_range[1], 3.05, 2.0], (300, 3))
    street = rng.uniform([x_range[0] - 1, 4.0, 0.0], [x_range[1] + 1, 5.0, 3.0], (200, 3))
    return np.vstack([glass, street])


def test_an_opening_is_set_in_its_wall() -> None:
    points = window_points((1.0, 2.0), np.random.default_rng(0))
    window = SceneObject(7, "window", box((1.5, 4.0, 1.5), (3.0, 2.0, 3.0), yaw=0.6))
    snapped = snap_opening(window, points, WALLS)
    assert snapped is not None
    assert snapped.box.center[:2] == pytest.approx([1.5, 3.0], abs=0.05)
    assert snapped.box.size == pytest.approx([1.0, 0.1, 1.0], abs=0.06)
    assert math.cos(snapped.box.yaw) == pytest.approx(1.0) or math.cos(snapped.box.yaw) == \
        pytest.approx(-1.0)


def test_duplicate_openings_merge_and_slivers_go() -> None:
    rng = np.random.default_rng(1)
    door = SceneObject(1, "door", box((1.5, 3.0, 1.5), (1.0, 0.2, 1.0)), movable=False)
    glazing = SceneObject(2, "window", box((1.6, 3.0, 1.5), (0.8, 0.2, 1.0)), movable=False)
    sliver = SceneObject(3, "window", box((3.5, 3.0, 1.5), (0.1, 0.2, 1.0)), movable=False)
    chair = SceneObject(4, "chair", box((2.0, 1.0, 0.45), (0.5, 0.5, 0.9)))
    points = {1: window_points((1.0, 2.0), rng), 2: window_points((1.2, 2.0), rng)[:200],
              3: window_points((3.45, 3.55), rng), 4: np.zeros((0, 3))}
    cleaned = clean_openings([door, glazing, sliver, chair], points, WALLS)
    assert [o.object_id for o in cleaned] == [4, 1]
    assert cleaned[1].box.size[0] == pytest.approx(1.0, abs=0.06)


def test_an_inner_door_goes_into_the_partition_between_two_rooms() -> None:
    rng = np.random.default_rng(0)
    floor = two_rooms_floor(0.9)  # The partition spans x in [4.0, 4.1].
    rows, cols = np.nonzero(floor.floor)
    floor_xy = floor.origin + (np.column_stack([cols, rows]) + 0.5) * floor.resolution
    walls = Walls.from_floor(floor_xy, 0.0, floor)
    # Depth seen through the door spreads its points on both sides of the partition.
    points = np.column_stack([rng.uniform(3.4, 4.7, 600), rng.uniform(1.6, 2.4, 600),
                              rng.uniform(0.0, 2.0, 600)])
    door = SceneObject(9, "door", box((4.0, 2.0, 1.0), (1.3, 0.8, 2.0)), movable=False)
    snapped = snap_opening(door, points, walls)
    assert snapped is not None
    assert snapped.box.center[:2] == pytest.approx([4.05, 2.0], abs=0.08)
    assert snapped.box.size[1] == pytest.approx(0.1)  # Thin, across the partition.
    # A door in an outer wall still goes into that wall.
    outer = np.column_stack([rng.uniform(0.0, 0.4, 400), rng.uniform(0.6, 1.4, 400),
                             rng.uniform(0.0, 2.0, 400)])
    front = SceneObject(8, "door", box((0.2, 1.0, 1.0), (0.4, 0.8, 2.0)), movable=False)
    snapped_front = snap_opening(front, outer, walls)
    assert snapped_front is not None and snapped_front.box.center[0] == pytest.approx(0.0, abs=0.1)
