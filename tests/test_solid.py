"""Tests of collisions between objects with voxel shapes."""

import numpy as np
import pytest

from pivot.geometry.collision import box_penetration
from pivot.geometry.shape import VoxelShape
from pivot.geometry.solid import solid_distance, solid_penetration
from pivot.scene.examples import box

RES = 0.04
TABLE_BOX = box((0.0, 0.0, 0.38), (1.2, 0.8, 0.76))


def table_shape() -> VoxelShape:
    """A table top (top 8 cm of the box) on four 8 cm legs."""
    i, j, k = np.meshgrid(np.arange(-15, 15), np.arange(-10, 10), np.arange(-10, 10), indexing="ij")
    cells = np.column_stack([i.ravel(), j.ravel(), k.ravel()])
    top = cells[:, 2] >= 7
    legs = (np.abs(cells[:, 0] + 0.5) >= 13) & (np.abs(cells[:, 1] + 0.5) >= 8)
    return VoxelShape(RES, cells[top | legs])


def test_membership_of_cells() -> None:
    shape = VoxelShape(RES, np.array([[0, 0, 0], [-3, 2, 1]]))
    assert shape.contains_cells(np.array([[-3, 2, 1], [0, 0, 1], [0, 0, 0]])).tolist() == \
        [True, False, True]


def test_a_carton_fits_under_a_table() -> None:
    carton = box((0.0, 0.0, 0.2), (0.4, 0.4, 0.4))
    assert box_penetration(carton, TABLE_BOX) > 0.3  # Boxes alone forbid it.
    assert solid_penetration(carton, None, TABLE_BOX, table_shape()) == 0.0
    assert solid_distance(carton, None, TABLE_BOX, table_shape()) == pytest.approx(0.28, abs=0.03)


def test_a_carton_hits_a_table_leg() -> None:
    carton = box((0.55, 0.35, 0.2), (0.4, 0.4, 0.4))
    assert solid_penetration(carton, None, TABLE_BOX, table_shape()) > 0.05
    assert solid_distance(carton, None, TABLE_BOX, table_shape()) == 0.0


def test_two_shapes_touching_do_not_collide() -> None:
    # Two cubes of cells side by side on offset grids: contact, not overlap.
    cube = VoxelShape(RES, np.stack(np.meshgrid(*[np.arange(-5, 5)] * 3), -1).reshape(-1, 3))
    left = box((0.0, 0.0, 0.2), (0.4, 0.4, 0.4))
    right = box((0.405, 0.01, 0.2), (0.4, 0.4, 0.4))
    assert solid_penetration(left, cube, right, cube) == 0.0
    deeper = box((0.3, 0.01, 0.2), (0.4, 0.4, 0.4))
    assert solid_penetration(left, cube, deeper, cube) == pytest.approx(0.1, abs=0.04)
