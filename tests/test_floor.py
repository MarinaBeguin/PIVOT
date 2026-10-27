"""Tests for the walkable floor map."""

import numpy as np
import pytest

from pivot.geometry.box import OrientedBox
from pivot.geometry.floor import FloorMap


def l_shaped_floor() -> FloorMap:
    """Builds an L-shaped 4 x 4 m room missing its top-right 2 x 2 m quadrant."""
    xs, ys = np.meshgrid(np.arange(0.0, 4.0, 0.02), np.arange(0.0, 4.0, 0.02))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    points = points[~((points[:, 0] > 2.0) & (points[:, 1] > 2.0))]
    return FloorMap.from_points(points, resolution=0.05, close_radius=0)


def footprint(x: float, y: float, size: float = 0.6) -> np.ndarray:
    """Returns the footprint of a square box centered at (x, y)."""
    return OrientedBox(np.array([x, y, 0.5]), np.array([size, size, 1.0])).footprint()


def test_footprint_inside_room_is_on_floor() -> None:
    assert l_shaped_floor().floor_fraction(footprint(1.0, 1.0)) == pytest.approx(1.0)


def test_footprint_in_missing_quadrant_is_off_floor() -> None:
    assert l_shaped_floor().floor_fraction(footprint(3.0, 3.0)) == pytest.approx(0.0)


def test_footprint_straddling_boundary_is_partial() -> None:
    fraction = l_shaped_floor().floor_fraction(footprint(2.0, 3.0))
    assert fraction == pytest.approx(0.5, abs=0.1)


def test_footprint_outside_grid_counts_as_off_floor() -> None:
    fraction = l_shaped_floor().floor_fraction(footprint(-0.1, 1.0, size=1.0))
    assert fraction < 0.75


def test_closing_fills_small_holes() -> None:
    xs, ys = np.meshgrid(np.arange(0.0, 2.0, 0.02), np.arange(0.0, 2.0, 0.02))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    # A 10 cm gap across the room, e.g. a texture-less strip.
    points = points[np.abs(points[:, 0] - 1.0) > 0.05]
    floor = FloorMap.from_points(points, resolution=0.05, close_radius=2)
    assert floor.is_floor(np.array([1.0, 1.0]))


def test_round_trip_serialization() -> None:
    floor = l_shaped_floor()
    restored = FloorMap.from_dict(floor.to_dict())
    np.testing.assert_array_equal(restored.floor, floor.floor)
    np.testing.assert_allclose(restored.origin, floor.origin)


def test_empty_points_are_rejected() -> None:
    with pytest.raises(ValueError):
        FloorMap.from_points(np.zeros((0, 2)))
