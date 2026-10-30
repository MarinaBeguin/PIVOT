"""Discretized configuration space of one object moving over the floor.

The object's configuration is ``(x, y, yaw)``. Positions are snapped to the
cells of the scene's floor map and yaws to ``yaw_bins`` values over half a
turn: a rectangle is symmetric under a rotation by pi.

For each yaw bin, the free configurations are computed for all cells at once
by correlating the obstacle and floor grids with the rasterized footprint.
The grid is an approximation; plans found in it are re-validated with exact
geometry by :mod:`pivot.planning.motion`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pivot.geometry.box import OrientedBox, rotation_2d, wrap_angle
from pivot.geometry.floor import FloorMap
from pivot.geometry.types import BoolArray, FloatArray
from pivot.planning.layers import free_of_obstacles, mover_stamps, obstacle_slabs
from pivot.scene.graph import Scene, carried_objects, is_opening

State = tuple[int, int, int]
"""A lattice configuration ``(yaw_bin, row, col)``."""

SLAB_HEIGHT = 0.10
"""Height of the slabs obstacles are tested in when objects have voxel shapes, in meters."""


def footprint_kernel(size_xy: FloatArray, yaw: float, resolution: float) -> BoolArray:
    """Rasterizes a centered rectangle footprint onto cell offsets.

    Args:
        size_xy: (2,) side lengths of the rectangle, in meters.
        yaw: Its orientation, in radians.
        resolution: Cell size, in meters.

    Returns:
        A square (2R + 1, 2R + 1) mask; entry ``[R + i, R + j]`` is True when
        the cell offset by ``i`` rows and ``j`` columns from the center cell
        has its center inside the rectangle.
    """
    radius = int(math.ceil(0.5 * float(np.hypot(*size_xy)) / resolution))
    offsets = np.arange(-radius, radius + 1) * resolution
    grid_x, grid_y = np.meshgrid(offsets, offsets)
    local = np.stack([grid_x, grid_y], axis=-1) @ rotation_2d(yaw)
    result: BoolArray = np.all(np.abs(local) <= 0.5 * size_xy + 1e-9, axis=-1)
    return result


def obstacle_grid(scene: Scene, object_id: int, floor: FloorMap, clearance: float,
                  tolerance: float) -> BoolArray:
    """Rasterizes the obstacles a moving object can hit, conservatively.

    The moving object, everything it carries, and objects entirely above or
    below its height range (including what it carries) by more than the
    clearance are not obstacles. Obstacle footprints are inflated by the
    clearance plus half a cell diagonal: two rectangles can overlap by that
    much without sharing a cell center, so the inflation keeps lattice
    configurations from being wrongly marked free.

    Args:
        scene: The scene.
        object_id: The moving object.
        floor: Grid to rasterize onto.
        clearance: Required distance to obstacles, in meters.
        tolerance: Interpenetration tolerated by exact placement checks.

    Returns:
        A (rows, cols) mask of blocked cells.
    """
    movers = [object_id, *carried_objects(scene, object_id)]
    z_lo = min(scene[i].box.z_min for i in movers)
    z_hi = max(scene[i].box.z_max for i in movers)
    margin = clearance + 0.5 * math.sqrt(2.0) * floor.resolution - tolerance
    blocked = np.zeros(floor.shape, dtype=bool)
    for other in scene.objects.values():
        if other.object_id in movers or is_opening(other):
            continue
        vertical_gap = max(other.box.z_min - z_hi, z_lo - other.box.z_max)
        if vertical_gap >= clearance - tolerance:
            continue
        size = other.box.size.copy()
        size[:2] = np.maximum(size[:2] + 2.0 * margin, 0.0)
        blocked |= floor.rasterize(OrientedBox(other.box.center, size, other.box.yaw).footprint())
    return blocked


@dataclass(frozen=True)
class ConfigurationSpace:
    """Free lattice configurations of one object.

    Attributes:
        floor: The floor map defining the position lattice.
        base_yaw: Yaw of bin 0, in radians (the object's current yaw).
        yaw_bins: Number of yaw bins over half a turn.
        free: (yaw_bins, rows, cols) mask of collision-free configurations.
        roomy: (yaw_bins, rows, cols) mask of configurations that stay free
            with an extra comfort margin; the planner prefers them.
    """

    floor: FloorMap
    base_yaw: float
    yaw_bins: int
    free: BoolArray
    roomy: BoolArray

    @classmethod
    def build(cls, scene: Scene, object_id: int, yaw_bins: int = 8, clearance: float = 0.0,
              floor_margin: float = 0.025, tolerance: float = 0.01,
              comfort: float = 0.08) -> ConfigurationSpace:
        """Computes the free configurations of an object in a scene.

        Args:
            scene: The scene. Must have a floor map.
            object_id: The moving object.
            yaw_bins: Number of yaw bins over half a turn.
            clearance: Required distance to other objects and walls, in meters.
            floor_margin: The footprint shrunk by this distance must lie on floor.
            tolerance: Interpenetration tolerated with other objects, in meters.
            comfort: Extra margin, beyond ``clearance``, of roomy configurations.

        Returns:
            The configuration space.

        Raises:
            ValueError: If the scene has no floor map.
        """
        if scene.floor is None:
            raise ValueError("Motion planning needs a floor map.")
        floor = scene.floor
        free = _free_configurations(scene, object_id, floor, yaw_bins, clearance,
                                    floor_margin, tolerance)
        roomy = free & _free_configurations(scene, object_id, floor, yaw_bins,
                                            clearance + comfort, floor_margin, tolerance)
        return cls(floor, scene[object_id].box.yaw, yaw_bins, free, roomy)

    def yaw(self, yaw_bin: int) -> float:
        """Returns the yaw of a bin.

        Args:
            yaw_bin: Bin index.

        Returns:
            The yaw in radians, wrapped to [-pi, pi).
        """
        return wrap_angle(self.base_yaw + yaw_bin * math.pi / self.yaw_bins)

    def nearest_bin(self, yaw: float) -> int:
        """Returns the bin closest to a yaw, modulo half a turn.

        Args:
            yaw: Yaw in radians.

        Returns:
            Bin index.
        """
        step = math.pi / self.yaw_bins
        return int(round(((yaw - self.base_yaw) % math.pi) / step)) % self.yaw_bins

    def position(self, row: int, col: int) -> FloatArray:
        """Returns the world xy of a cell center.

        Args:
            row: Cell row.
            col: Cell column.

        Returns:
            The (2,) position.
        """
        cell = np.array([col, row]) + 0.5
        result: FloatArray = self.floor.origin + cell * self.floor.resolution
        return result

    def nearest_cell(self, point_xy: FloatArray) -> tuple[int, int]:
        """Returns the cell containing a point, clamped to the grid.

        Args:
            point_xy: (2,) world xy.

        Returns:
            ``(row, col)``.
        """
        col, row = np.floor((point_xy - self.floor.origin) / self.floor.resolution).astype(int)
        rows, cols = self.floor.shape
        return int(np.clip(row, 0, rows - 1)), int(np.clip(col, 0, cols - 1))

    def free_centers(self, yaw_bin: int) -> FloatArray:
        """Returns the world xy of the free cells of a yaw bin.

        Args:
            yaw_bin: Bin index.

        Returns:
            (N, 2) positions.
        """
        rows, cols = np.nonzero(self.free[yaw_bin])
        result: FloatArray = self.floor.origin + (np.column_stack([cols, rows]) + 0.5) \
            * self.floor.resolution
        return result

    def is_free(self, state: State) -> bool:
        """Tests whether a lattice state is free.

        Args:
            state: ``(yaw_bin, row, col)``.

        Returns:
            True if the state is inside the grid and collision-free.
        """
        k, row, col = state
        rows, cols = self.floor.shape
        return 0 <= row < rows and 0 <= col < cols and bool(self.free[k, row, col])

    def box_at(self, template: OrientedBox, state: State) -> OrientedBox:
        """Places an object's box at a lattice state.

        Args:
            template: The object's box, providing size and height.
            state: ``(yaw_bin, row, col)``.

        Returns:
            The placed box.
        """
        k, row, col = state
        return template.moved(self.position(row, col), self.yaw(k))


def _free_configurations(scene: Scene, object_id: int, floor: FloorMap, yaw_bins: int,
                         margin: float, floor_margin: float, tolerance: float) -> BoolArray:
    """Computes the configurations keeping a margin to obstacles and walls.

    When any object has a voxel shape, obstacles are tested slab by slab
    (:mod:`pivot.planning.layers`); otherwise on box footprints.

    Args:
        scene: The scene.
        object_id: The moving object.
        floor: The scene's floor map.
        yaw_bins: Number of yaw bins over half a turn.
        margin: Required distance to obstacles and walls, in meters.
        floor_margin: The footprint shrunk by this distance must lie on floor.
        tolerance: Interpenetration tolerated with other objects, in meters.

    Returns:
        A (yaw_bins, rows, cols) mask.
    """
    box = scene[object_id].box
    yaws = [box.yaw + k * math.pi / yaw_bins for k in range(yaw_bins)]
    if any(o.shape is not None for o in scene.objects.values()):
        unblocked = _unblocked_by_slabs(scene, object_id, floor, yaws, margin, tolerance)
    else:
        blocked = obstacle_grid(scene, object_id, floor, margin, tolerance)
        unblocked = [_unblocked(blocked, box.size[:2], yaw, floor.resolution) for yaw in yaws]
    mask: BoolArray = np.stack([
        free & _on_floor(floor, box.size[:2], yaw, floor_margin - margin)
        for free, yaw in zip(unblocked, yaws)
    ])
    return mask


def _unblocked_by_slabs(scene: Scene, object_id: int, floor: FloorMap, yaws: list[float],
                        margin: float, tolerance: float) -> list[BoolArray]:
    """Computes, for each yaw, the positions where an object hits nothing, slab by slab.

    Args:
        scene: The scene.
        object_id: The moving object.
        floor: The scene's floor map.
        yaws: Orientations to test, in radians.
        margin: Required distance to obstacles, in meters.
        tolerance: Interpenetration tolerated with other objects, in meters.

    Returns:
        One (rows, cols) mask per yaw.
    """
    movers = [object_id, *carried_objects(scene, object_id)]
    passengers = [scene[i] for i in movers[1:]]
    stamps = [mover_stamps(scene[object_id], yaw, floor.resolution, SLAB_HEIGHT, passengers)
              for yaw in yaws]
    slabs = sorted(set().union(*stamps))
    obstacles = obstacle_slabs(scene, movers, floor, slabs, SLAB_HEIGHT, margin, tolerance)
    return [free_of_obstacles(stamp, obstacles) for stamp in stamps]


def _unblocked(blocked: BoolArray, size_xy: FloatArray, yaw: float,
               resolution: float) -> BoolArray:
    """Computes the positions where a footprint covers no blocked cell.

    Args:
        blocked: (rows, cols) obstacle mask.
        size_xy: (2,) footprint side lengths, in meters.
        yaw: Footprint orientation, in radians.
        resolution: Cell size, in meters.

    Returns:
        A (rows, cols) mask of free center cells.
    """
    kernel = footprint_kernel(size_xy, yaw, resolution).astype(np.int32)
    # Cells outside the grid are not obstacles.
    hits = ndimage.correlate(blocked.astype(np.int32), kernel, mode="constant", cval=0)
    result: BoolArray = hits == 0
    return result


def _on_floor(floor: FloorMap, size_xy: FloatArray, yaw: float, floor_margin: float) -> BoolArray:
    """Computes the positions where a footprint stands on walkable floor.

    Args:
        floor: The floor map.
        size_xy: (2,) footprint side lengths, in meters.
        yaw: Footprint orientation, in radians.
        floor_margin: The footprint shrunk by this distance must lie on floor;
            a negative margin grows it (wall clearance).

    Returns:
        A (rows, cols) mask of valid center cells.
    """
    inner_size = np.maximum(size_xy - 2.0 * floor_margin, 0.0)
    inner = footprint_kernel(inner_size, yaw, floor.resolution).astype(np.int32)
    # Cells outside the grid are not floor.
    on_floor = ndimage.correlate(floor.floor.astype(np.int32), inner, mode="constant", cval=0)
    result: BoolArray = on_floor == inner.sum()
    return result
