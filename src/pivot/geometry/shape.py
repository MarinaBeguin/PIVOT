"""Voxel shapes: the actual volume of an object inside its bounding box.

A box is a poor model of most furniture: a table is mostly empty space under
its top, a chair mostly air between its legs. A :class:`VoxelShape` lists the
occupied cells of a regular grid attached to the object's box, so it follows
the box when the object moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.geometry.types import BoolArray, FloatArray, IntArray

# Cell indices are packed into one integer, 21 bits per axis (about +-40 km at 4 cm).
_OFFSET = 1 << 20


def _cell_keys(cells: IntArray) -> IntArray:
    """Packs (N, 3) cell indices into sortable scalar keys.

    Args:
        cells: (N, 3) cell indices.

    Returns:
        (N,) keys.
    """
    shifted = cells + _OFFSET
    result: IntArray = (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]
    return result


@dataclass(frozen=True)
class VoxelShape:
    """Occupied cells of a grid in an object's box frame.

    The box frame has its origin at the box center, x along the box yaw and z
    up. Cell ``(i, j, k)`` spans ``[i, i + 1] * resolution`` along x, and so on.

    Attributes:
        resolution: Cell side length, in meters.
        cells: (M, 3) integer indices of the occupied cells.
    """

    resolution: float
    cells: IntArray

    def __post_init__(self) -> None:
        cells = np.unique(np.asarray(self.cells, dtype=np.int64).reshape(-1, 3), axis=0)
        object.__setattr__(self, "cells", cells)

    def __len__(self) -> int:
        return len(self.cells)

    @cached_property
    def _keys(self) -> IntArray:
        """Sorted scalar keys of the occupied cells, for fast membership tests."""
        return _cell_keys(self.cells)

    def contains_cells(self, cells: IntArray) -> BoolArray:
        """Tests which cells are occupied.

        Args:
            cells: (N, 3) cell indices.

        Returns:
            (N,) mask.
        """
        keys = _cell_keys(np.asarray(cells, dtype=np.int64).reshape(-1, 3))
        if len(self._keys) == 0:
            return np.zeros(len(keys), dtype=bool)
        positions = np.clip(np.searchsorted(self._keys, keys), 0, len(self._keys) - 1)
        result: BoolArray = self._keys[positions] == keys
        return result

    def local_centers(self) -> FloatArray:
        """Returns the cell centers in the box frame.

        Returns:
            (M, 3) points, in meters.
        """
        result: FloatArray = (self.cells + 0.5) * self.resolution
        return result

    def world_centers(self, box: OrientedBox) -> FloatArray:
        """Returns the cell centers for the object at a given pose.

        Args:
            box: The object's box at that pose.

        Returns:
            (M, 3) world points.
        """
        local = self.local_centers()
        xy = local[:, :2] @ rotation_2d(box.yaw).T + box.center[:2]
        result: FloatArray = np.column_stack([xy, local[:, 2] + box.center[2]])
        return result

    @classmethod
    def from_world_points(cls, box: OrientedBox, points: FloatArray,
                          resolution: float) -> VoxelShape:
        """Builds the shape occupying the cells that contain given points.

        Args:
            box: The object's box.
            points: (N, 3) world points inside the object.
            resolution: Cell side length, in meters.

        Returns:
            The shape.
        """
        return cls(resolution, np.floor(box.to_local(points) / resolution).astype(np.int64))

    def to_dict(self) -> dict[str, Any]:
        """Serializes the shape to JSON-compatible types.

        Returns:
            A dictionary representation.
        """
        return {"resolution": self.resolution, "cells": self.cells.ravel().tolist()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoxelShape:
        """Builds a shape from the output of :meth:`to_dict`.

        Args:
            data: Serialized shape.

        Returns:
            The deserialized shape.
        """
        return cls(float(data["resolution"]), np.array(data["cells"], dtype=np.int64))
