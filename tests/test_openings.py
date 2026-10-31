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
    door = SceneObject(1, "window", box((1.5, 3.0, 1.5), (1.0, 0.2, 1.0)), movable=False)
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


def test_pieces_of_one_door_are_joined() -> None:
    rng = np.random.default_rng(2)
    # The top and bottom of a door in the wall y = 3, segmented apart, its middle missed.
    top = rng.uniform([1.0, 2.95, 1.4], [1.9, 3.05, 2.0], (300, 3))
    bottom = rng.uniform([1.0, 2.95, 0.0], [1.9, 3.05, 0.8], (200, 3))
    doors = [SceneObject(1, "door", box((1.45, 3.0, 1.55), (0.9, 0.1, 0.9)), movable=False),
             SceneObject(2, "door", box((1.45, 3.0, 0.5), (0.9, 0.1, 1.0)), movable=False)]
    cleaned = clean_openings(doors, {1: top, 2: bottom}, WALLS)
    assert [o.object_id for o in cleaned] == [1]
    assert cleaned[0].box.z_min == pytest.approx(0.0, abs=0.05)
    assert cleaned[0].box.z_max == pytest.approx(2.0, abs=0.05)


def test_a_mirror_hangs_flat_on_its_wall_and_a_curtain_in_mid_room_goes() -> None:
    rng = np.random.default_rng(3)
    # A mirror on the wall x = 4: its frame, and the room it shows, 2 m behind the wall.
    frame = rng.uniform([3.95, 1.0, 1.0], [4.05, 2.0, 2.0], (300, 3))
    reflection = rng.uniform([5.0, 0.0, 0.0], [6.5, 3.0, 2.5], (600, 3))
    mirror = SceneObject(5, "mirror", box((5.0, 1.5, 1.25), (3.0, 3.0, 2.5)))
    curtain = SceneObject(6, "curtain", box((2.0, 1.5, 1.1), (1.0, 1.0, 2.2)))
    middle = rng.uniform([1.5, 1.0, 0.0], [2.5, 2.0, 2.2], (300, 3))
    cleaned = clean_openings([mirror, curtain], {5: np.vstack([frame, reflection]),
                                                 6: middle}, WALLS)
    assert [o.object_id for o in cleaned] == [5]
    hung = cleaned[0]
    assert hung.movable  # A mirror can be moved; a curtain cannot.
    assert hung.box.center[:2] == pytest.approx([4.0, 1.5], abs=0.05)
    assert hung.box.size[1] == pytest.approx(0.05)
    assert hung.box.size[0] == pytest.approx(1.0, abs=0.06)


def test_cabinet_doors_and_what_a_mirror_shows_are_dropped() -> None:
    rng = np.random.default_rng(4)
    floor = two_rooms_floor(0.9)  # Two rooms, x in [0, 8], the partition at x = 4.
    rows, cols = np.nonzero(floor.floor)
    floor_xy = floor.origin + (np.column_stack([cols, rows]) + 0.5) * floor.resolution
    walls = Walls.from_floor(floor_xy, 0.0, floor)
    # A cabinet's door, 40 cm tall, against the wall y = 0.
    cabinet = SceneObject(1, "door", box((1.0, 0.2, 0.25), (0.5, 0.1, 0.4)), movable=False)
    low = rng.uniform([0.75, 0.0, 0.05], [1.25, 0.1, 0.45], (200, 3))
    # A mirror seen in another mirror: on a wall 2 m outside the home.
    far = SceneObject(2, "mirror", box((10.0, 2.0, 1.4), (0.8, 0.1, 1.0)))
    ghost = rng.uniform([9.95, 1.6, 0.9], [10.05, 2.4, 1.9], (300, 3))
    cleaned = clean_openings([cabinet, far], {1: low, 2: ghost},
                             Walls(walls.yaw, walls.low, np.array([10.0, walls.high[1]]),
                                   walls.floor))
    assert not cleaned


def test_a_curtain_by_its_window_is_fixed() -> None:
    rng = np.random.default_rng(5)
    drape = rng.uniform([0.0, 1.0, 0.1], [0.3, 2.0, 2.3], (400, 3))
    curtain = SceneObject(7, "curtain", box((0.15, 1.5, 1.2), (0.3, 1.0, 2.2)))
    cleaned = clean_openings([curtain], {7: drape}, WALLS)
    assert len(cleaned) == 1 and not cleaned[0].movable
    assert cleaned[0].box.size[1] == pytest.approx(0.15)
