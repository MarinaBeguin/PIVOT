"""2D floor map: where in the room something can stand.

The map is a boolean grid over the floor plane. A cell is walkable floor if
floor points were observed in it. Footprints of furniture are rasterized onto
this grid to check that a proposed placement stays inside the room.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from pivot.geometry.collision import points_to_polygon_distance
from pivot.geometry.types import BoolArray, FloatArray, IntArray


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

    def cells_inside(self, polygon: FloatArray) -> tuple[IntArray, IntArray]:
        """Lists the cells whose centers lie inside a convex polygon.

        Cells beyond the grid bounds are included, so callers can tell a
        footprint that leaves the mapped area.

        Args:
            polygon: (N, 2) convex polygon, counter-clockwise.

        Returns:
            A tuple ``(rows, cols)`` of cell indices, possibly out of bounds.
        """
        lo = np.floor((polygon.min(axis=0) - self.origin) / self.resolution).astype(int)
        hi = np.ceil((polygon.max(axis=0) - self.origin) / self.resolution).astype(int)
        cols, rows = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
        centers = self.origin + (np.stack([cols, rows], axis=-1) + 0.5) * self.resolution
        inside = np.ones(cols.shape, dtype=bool)
        for i, start in enumerate(polygon):
            edge = polygon[(i + 1) % len(polygon)] - start
            rel = centers - start
            # Left of every counter-clockwise edge means inside.
            inside &= edge[0] * rel[..., 1] - edge[1] * rel[..., 0] >= 0.0
        return rows[inside], cols[inside]

    def _in_bounds(self, rows: IntArray, cols: IntArray) -> BoolArray:
        """Tests which cell indices fall inside the grid.

        Args:
            rows: Row indices.
            cols: Column indices.

        Returns:
            A mask of in-bounds cells.
        """
        n_rows, n_cols = self.shape
        result: BoolArray = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)
        return result

    def rasterize(self, polygon: FloatArray) -> BoolArray:
        """Marks the grid cells whose centers lie inside a convex polygon.

        Args:
            polygon: (N, 2) convex polygon, counter-clockwise.

        Returns:
            A (rows, cols) mask.
        """
        rows, cols = self.cells_inside(polygon)
        keep = self._in_bounds(rows, cols)
        mask = np.zeros(self.shape, dtype=bool)
        mask[rows[keep], cols[keep]] = True
        return mask

    def floor_fraction(self, polygon: FloatArray) -> float:
        """Computes the fraction of a footprint that stands on walkable floor.

        Footprint cells outside the grid count as off the floor.

        Args:
            polygon: (N, 2) convex polygon, counter-clockwise.

        Returns:
            A value in [0, 1]. Footprints containing no cell center are tested
            at their centroid.
        """
        rows, cols = self.cells_inside(polygon)
        if len(rows) == 0:
            return 1.0 if self.is_floor(polygon.mean(axis=0)) else 0.0
        keep = self._in_bounds(rows, cols)
        on_floor = int(self.floor[rows[keep], cols[keep]].sum())
        return on_floor / len(rows)

    def distance_to_walls(self, polygon: FloatArray, search_radius: float = 1.0) -> float:
        """Measures the gap between a footprint and the nearest non-floor cell.

        Non-floor cells (walls, unobserved areas, outside the map) are treated as
        solid squares.

        Args:
            polygon: (N, 2) convex footprint, counter-clockwise.
            search_radius: Gaps larger than this are reported as infinite.

        Returns:
            The gap in meters; 0 if the footprint overlaps a non-floor cell.
        """
        lo = polygon.min(axis=0) - search_radius
        hi = polygon.max(axis=0) + search_radius
        start = np.floor((lo - self.origin) / self.resolution).astype(int)
        stop = np.ceil((hi - self.origin) / self.resolution).astype(int)
        cols, rows = np.meshgrid(np.arange(start[0], stop[0] + 1),
                                 np.arange(start[1], stop[1] + 1))
        rows, cols = rows.ravel(), cols.ravel()
        solid = ~self._in_bounds(rows, cols)
        keep = ~solid
        solid[keep] = ~self.floor[rows[keep], cols[keep]]
        if not solid.any():
            return math.inf
        cells = np.column_stack([cols[solid], rows[solid]]) + 0.5
        centers = self.origin + cells * self.resolution
        gaps = points_to_polygon_distance(centers, polygon) - 0.5 * self.resolution
        nearest = float(max(0.0, gaps.min()))
        return nearest if nearest <= search_radius else math.inf

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
