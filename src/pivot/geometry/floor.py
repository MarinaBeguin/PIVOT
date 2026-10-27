"""2D floor map: where in the room something can stand.

The map is a boolean grid over the floor plane. A cell is walkable floor if
floor points were observed in it. Footprints of furniture are rasterized onto
this grid to check that a proposed placement stays inside the room.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from pivot.geometry.types import BoolArray, FloatArray


@dataclass(frozen=True)
class FloorMap:
    """Boolean occupancy grid of the observed floor.

    Cell ``(row, col)`` covers ``origin + [col, row] * resolution`` to
    ``origin + [col + 1, row + 1] * resolution``: rows follow +y, columns +x.

    Attributes:
        origin: (2,) world xy of the grid's lower-left corner, in meters.
        resolution: Cell side length, in meters.
        floor: (rows, cols) mask, True where the floor is walkable.
    """

    origin: FloatArray
    resolution: float
    floor: BoolArray

    @classmethod
    def from_points(cls, points_xy: FloatArray, resolution: float = 0.05,
                    close_radius: int = 2, min_points: int = 1) -> FloorMap:
        """Builds a floor map from observed floor points.

        Reconstructed floors have holes (occlusions, texture-less areas). A
        morphological closing fills holes up to ``close_radius`` cells wide,
        and interior holes enclosed by floor are filled entirely.

        Args:
            points_xy: (N, 2) xy coordinates of floor points.
            resolution: Cell side length, in meters.
            close_radius: Radius of the closing structuring element, in cells.
            min_points: Minimum number of points for a cell to count as floor.

        Returns:
            The floor map.

        Raises:
            ValueError: If ``points_xy`` is empty.
        """
        points_xy = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        if len(points_xy) == 0:
            raise ValueError("Cannot build a floor map without floor points.")
        pad = (close_radius + 1) * resolution
        origin = points_xy.min(axis=0) - pad
        shape = np.ceil((points_xy.max(axis=0) + pad - origin) / resolution).astype(int)
        cols, rows = shape
        index = np.floor((points_xy - origin) / resolution).astype(int)
        counts = np.zeros((rows, cols), dtype=np.int64)
        np.add.at(counts, (index[:, 1], index[:, 0]), 1)
        floor = counts >= min_points
        if close_radius > 0:
            structure = ndimage.generate_binary_structure(2, 1)
            floor = ndimage.binary_closing(floor, structure, iterations=close_radius)
        floor = ndimage.binary_fill_holes(floor)
        return cls(origin, resolution, floor)

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape as ``(rows, cols)``."""
        rows, cols = self.floor.shape
        return int(rows), int(cols)

    def cell_centers(self) -> FloatArray:
        """Returns the world xy of every cell center.

        Returns:
            A (rows, cols, 2) array.
        """
        rows, cols = self.shape
        xs = self.origin[0] + (np.arange(cols) + 0.5) * self.resolution
        ys = self.origin[1] + (np.arange(rows) + 0.5) * self.resolution
        grid_x, grid_y = np.meshgrid(xs, ys)
        result: FloatArray = np.stack([grid_x, grid_y], axis=-1)
        return result

    def rasterize(self, polygon: FloatArray) -> BoolArray:
        """Marks the cells whose centers lie inside a convex polygon.

        Args:
            polygon: (N, 2) convex polygon, counter-clockwise.

        Returns:
            A (rows, cols) mask.
        """
        centers = self.cell_centers()
        inside = np.ones(self.shape, dtype=bool)
        for i, start in enumerate(polygon):
            edge = polygon[(i + 1) % len(polygon)] - start
            rel = centers - start
            # Left of every counter-clockwise edge means inside.
            inside &= edge[0] * rel[..., 1] - edge[1] * rel[..., 0] >= 0.0
        return inside

    def floor_fraction(self, polygon: FloatArray) -> float:
        """Computes the fraction of a footprint that stands on walkable floor.

        Footprint parts outside the grid count as off the floor.

        Args:
            polygon: (N, 2) convex polygon, counter-clockwise.

        Returns:
            A value in [0, 1]. Footprints smaller than a cell are tested at their
            centroid.
        """
        mask = self.rasterize(polygon)
        total = int(mask.sum())
        if total == 0:
            return 1.0 if self.is_floor(polygon.mean(axis=0)) else 0.0
        area = _polygon_area(polygon) / self.resolution**2
        # Cells outside the grid are not in the mask; account for them through area.
        return min(1.0, float((mask & self.floor).sum()) / max(area, total))

    def is_floor(self, point_xy: FloatArray) -> bool:
        """Tests whether a single point lies on walkable floor.

        Args:
            point_xy: (2,) world xy.

        Returns:
            True if the point falls in a floor cell.
        """
        col, row = np.floor((np.asarray(point_xy) - self.origin) / self.resolution).astype(int)
        rows, cols = self.shape
        return bool(0 <= row < rows and 0 <= col < cols and self.floor[row, col])

    def to_dict(self) -> dict[str, Any]:
        """Serializes the map to JSON-compatible types.

        Returns:
            A dictionary with the origin, the resolution, the column count and,
            for each row, the column indices of its floor cells.
        """
        return {
            "origin": self.origin.tolist(),
            "resolution": self.resolution,
            "floor": [np.flatnonzero(row).tolist() for row in self.floor],
            "cols": self.shape[1],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FloorMap:
        """Builds a map from the output of :meth:`to_dict`.

        Args:
            data: Serialized map.

        Returns:
            The deserialized map.
        """
        floor = np.zeros((len(data["floor"]), int(data["cols"])), dtype=bool)
        for row, cols in enumerate(data["floor"]):
            floor[row, cols] = True
        return cls(np.asarray(data["origin"], dtype=np.float64), float(data["resolution"]), floor)


def _polygon_area(polygon: FloatArray) -> float:
    """Computes the area of a simple polygon with the shoelace formula.

    Args:
        polygon: (N, 2) vertices.

    Returns:
        The unsigned area.
    """
    x, y = polygon[:, 0], polygon[:, 1]
    return 0.5 * abs(float(x @ np.roll(y, -1) - y @ np.roll(x, -1)))
