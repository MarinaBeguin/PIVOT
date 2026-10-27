"""Tests for polygon and box distance and overlap queries."""

import math

import numpy as np
import pytest

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import (
    box_distance,
    box_penetration,
    boxes_collide,
    interval_gap,
    polygon_distance,
    polygon_penetration,
)


def unit_box(x: float, y: float, z: float = 0.5, yaw: float = 0.0) -> OrientedBox:
    """Returns a 1 m cube centered at the given position."""
    return OrientedBox(np.array([x, y, z]), np.ones(3), yaw)


def test_separated_boxes_have_axis_distance() -> None:
    a, b = unit_box(0.0, 0.0), unit_box(3.0, 0.0)
    assert box_distance(a, b) == pytest.approx(2.0)
    assert box_penetration(a, b) == 0.0
    assert not boxes_collide(a, b)


def test_diagonal_distance_uses_corners() -> None:
    a, b = unit_box(0.0, 0.0), unit_box(2.0, 2.0)
    assert box_distance(a, b) == pytest.approx(math.sqrt(2.0))


def test_rotated_box_distance() -> None:
    # A diamond whose corner points at the unit square, 1 m away.
    a = unit_box(0.0, 0.0)
    corner = math.sqrt(0.5)
    b = unit_box(0.5 + 1.0 + corner, 0.0, yaw=math.pi / 4)
    assert box_distance(a, b) == pytest.approx(1.0)


def test_overlapping_boxes_report_penetration() -> None:
    a, b = unit_box(0.0, 0.0), unit_box(0.7, 0.0)
    assert box_penetration(a, b) == pytest.approx(0.3)
    assert box_distance(a, b) == 0.0
    assert boxes_collide(a, b)


def test_touching_boxes_do_not_collide() -> None:
    assert not boxes_collide(unit_box(0.0, 0.0), unit_box(1.0, 0.0))


def test_vertical_separation_avoids_collision() -> None:
    low, high = unit_box(0.0, 0.0, z=0.5), unit_box(0.2, 0.0, z=2.0)
    assert box_penetration(low, high) == 0.0
    assert box_distance(low, high) == pytest.approx(0.5)


def test_clearance_turns_near_miss_into_conflict() -> None:
    a, b = unit_box(0.0, 0.0), unit_box(1.2, 0.0)
    assert not boxes_collide(a, b)
    assert boxes_collide(a, b, clearance=0.5)


def test_polygon_helpers_are_symmetric() -> None:
    first = unit_box(0.0, 0.0).footprint()
    second = unit_box(2.5, 1.0, yaw=0.4).footprint()
    assert polygon_distance(first, second) == pytest.approx(polygon_distance(second, first))
    assert polygon_penetration(first, second) == polygon_penetration(second, first) == 0.0


@pytest.mark.parametrize(("a", "b", "expected"), [
    ((0.0, 1.0), (2.0, 3.0), 1.0),
    ((2.0, 3.0), (0.0, 1.0), 1.0),
    ((0.0, 2.0), (1.0, 3.0), 0.0),
])
def test_interval_gap(a: tuple[float, float], b: tuple[float, float], expected: float) -> None:
    assert interval_gap(*a, *b) == pytest.approx(expected)
