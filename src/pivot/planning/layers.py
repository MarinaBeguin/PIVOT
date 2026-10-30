"""Free configurations of objects with voxel shapes, slab by slab.

A box footprint says nothing about height: a chair seat fits under a table
top, a carton between its legs. Space is cut into horizontal slabs; in each
slab, the moving object's cells form a footprint stamp and the other objects'
cells an obstacle map. A position is free when, in every slab the object
occupies, its stamp hits no obstacle. Objects without a shape fill their box
in the slabs it spans.

Stamps are cell centers snapped to the floor grid; obstacles are inflated by
the half extent of both objects' cells and half a grid diagonal, so that the
lattice errs towards blocked. Plans are re-validated exactly afterwards.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage, signal

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.geometry.floor import FloorMap
from pivot.geometry.types import BoolArray, FloatArray
from pivot.scene.graph import Scene, SceneObject, is_opening


def slab_range(z_min: float, z_max: float, height: float) -> range:
    """Lists the slabs a vertical interval touches.

    Args:
        z_min: Bottom of the interval, in meters.
        z_max: Top of the interval, in meters.
        height: Slab height, in meters.

    Returns:
        Slab indices; slab ``k`` spans ``[k, k + 1) * height``.
    """
    return range(int(math.floor(z_min / height)), int(math.floor(z_max / height)) + 1)


def _cell_radius(obj: SceneObject) -> float:
    """Returns the horizontal half diagonal of an object's cells, 0 without a shape.

    Args:
        obj: The object.

    Returns:
        The radius, in meters.
    """
    return 0.5 * math.sqrt(2.0) * obj.shape.resolution if obj.shape is not None else 0.0


def _stamp(points: FloatArray, resolution: float) -> BoolArray:
    """Rasterizes footprint offsets onto a centered square of grid cells.

    Args:
        points: (N, 2) offsets from the object's center, in meters.
        resolution: Grid cell size, in meters.

    Returns:
        A (2R + 1, 2R + 1) mask; entry ``[R + i, R + j]`` is True when a point
        falls in the cell offset by ``i`` rows and ``j`` columns.
    """
    cells = np.rint(points / resolution).astype(np.int64)
    radius = int(np.abs(cells).max(initial=0))
    stamp = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=bool)
    stamp[cells[:, 1] + radius, cells[:, 0] + radius] = True
    return stamp


def _solid_points(obj: SceneObject, spacing: float = 0.04) -> FloatArray:
    """Samples the volume an object occupies: its cell centers, or a grid filling its box.

    Args:
        obj: The object.
        spacing: Grid spacing for objects without a shape, in meters.

    Returns:
        (N, 3) world points.
    """
    if obj.shape is not None:
        return obj.shape.world_centers(obj.box)
    axes = [np.arange(-0.5 * s, 0.5 * s + 1e-9, spacing) if s > spacing else np.zeros(1)
            for s in obj.box.size]
    local = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    xy = local[:, :2] @ rotation_2d(obj.box.yaw).T + obj.box.center[:2]
    result: FloatArray = np.column_stack([xy, local[:, 2] + obj.box.center[2]])
    return result


def _merge(a: BoolArray, b: BoolArray) -> BoolArray:
    """Unites two centered stamps of possibly different sizes.

    Args:
        a: First stamp.
        b: Second stamp.

    Returns:
        A centered stamp covering both.
    """
    size = max(len(a), len(b))
    result = np.zeros((size, size), dtype=bool)
    for stamp in (a, b):
        offset = (size - len(stamp)) // 2
        result[offset:offset + len(stamp), offset:offset + len(stamp)] |= stamp
    return result


def mover_stamps(obj: SceneObject, yaw: float, resolution: float, height: float,
                 passengers: list[SceneObject] | None = None) -> dict[int, BoolArray]:
    """Computes the footprint stamp of a moving object in each slab it occupies.

    Args:
        obj: The moving object; its box gives the height, its shape the cells.
        yaw: Orientation of the object, in radians.
        resolution: Floor grid cell size, in meters.
        height: Slab height, in meters.
        passengers: Objects it carries, which turn with it about its center.

    Returns:
        Stamps keyed by slab index.
    """
    box = obj.box
    if obj.shape is None:
        stamp = _box_stamp(box.size[:2], yaw, resolution)
        stamps = {k: stamp for k in slab_range(box.z_min, box.z_max, height)}
    else:
        local = obj.shape.local_centers()
        stamps = _slab_stamps(local[:, :2] @ rotation_2d(yaw).T, local[:, 2] + box.center[2],
                              resolution, height)
    for passenger in passengers or []:
        points = _solid_points(passenger)
        # Offsets in the mover's frame, then turned to the tested yaw.
        offsets = (points[:, :2] - box.center[:2]) @ rotation_2d(box.yaw) @ rotation_2d(yaw).T
        for k, stamp in _slab_stamps(offsets, points[:, 2], resolution, height).items():
            stamps[k] = _merge(stamps[k], stamp) if k in stamps else stamp
    return stamps


def _box_stamp(size_xy: FloatArray, yaw: float, resolution: float) -> BoolArray:
    """Rasterizes a centered rectangle: the cells whose center it contains.

    Args:
        size_xy: (2,) side lengths, in meters.
        yaw: Orientation, in radians.
        resolution: Grid cell size, in meters.

    Returns:
        A centered square stamp.
    """
    reach = math.ceil(0.5 * math.hypot(*size_xy) / resolution)
    offsets = np.arange(-reach, reach + 1) * resolution
    grid = np.stack(np.meshgrid(offsets, offsets), axis=-1).reshape(-1, 2)
    inside = np.all(np.abs(grid @ rotation_2d(yaw)) <= 0.5 * size_xy + 1e-9, axis=1)
    return _stamp(grid[inside], resolution)


def _slab_stamps(offsets: FloatArray, heights: FloatArray, resolution: float,
                 height: float) -> dict[int, BoolArray]:
    """Rasterizes horizontal offsets into one stamp per slab.

    Args:
        offsets: (N, 2) offsets from the mover's center, in meters.
        heights: (N,) heights of the points, in meters.
        resolution: Floor grid cell size, in meters.
        height: Slab height, in meters.

    Returns:
        Stamps keyed by slab index.
    """
    slabs = np.floor(heights / height).astype(np.int64)
    return {int(k): _stamp(offsets[slabs == k], resolution) for k in np.unique(slabs)}


def _shape_obstacle(obj: SceneObject, floor: FloorMap, slab: int, height: float,
                    radius: float, band: tuple[float, float]) -> BoolArray:
    """Rasterizes the cells of a shaped object lying in one slab.

    Args:
        obj: The obstacle; must have a shape.
        floor: Grid to rasterize onto.
        slab: Slab index.
        height: Slab height, in meters.
        radius: Inflation of every cell center, in meters.
        band: Heights the moving object can reach; cells outside are ignored.

    Returns:
        A (rows, cols) mask.
    """
    assert obj.shape is not None
    centers = obj.shape.world_centers(obj.box)
    half = 0.5 * obj.shape.resolution
    # A cell belongs to every slab its vertical extent touches.
    inside = (centers[:, 2] + half > max(slab * height, band[0])) \
        & (centers[:, 2] - half < min((slab + 1) * height, band[1]))
    grid = np.zeros(floor.shape, dtype=bool)
    cells = np.floor((centers[inside, :2] - floor.origin) / floor.resolution).astype(np.int64)
    valid = ((cells >= 0) & (cells < np.array(floor.shape[::-1]))).all(axis=1)
    grid[cells[valid, 1], cells[valid, 0]] = True
    steps = int(math.ceil(radius / floor.resolution))
    disk = np.hypot(*np.mgrid[-steps:steps + 1, -steps:steps + 1]) * floor.resolution <= radius
    result: BoolArray = ndimage.binary_dilation(grid, structure=disk) if grid.any() else grid
    return result


def obstacle_slabs(scene: Scene, movers: list[int], floor: FloorMap, slabs: list[int],
                   height: float, margin: float, tolerance: float) -> dict[int, BoolArray]:
    """Rasterizes, slab by slab, everything a moving object can hit.

    Args:
        scene: The scene.
        movers: The moving object and everything it carries; never obstacles.
        floor: Grid to rasterize onto.
        slabs: Slabs the moving object occupies.
        height: Slab height, in meters.
        margin: Required distance to obstacles, in meters.
        tolerance: Interpenetration tolerated by exact placement checks, in meters.

    Returns:
        (rows, cols) obstacle masks keyed by slab.
    """
    # Inflation shared by all obstacles; each adds the half extent of its cells.
    inflation = (margin + 0.5 * math.sqrt(2.0) * floor.resolution - tolerance
                 + _cell_radius(scene[movers[0]]))
    # Clearance also applies vertically: obstacles reach into neighboring slabs.
    reach = int(math.ceil(max(margin, 0.0) / height))
    band = _height_band(scene, movers, margin, tolerance)
    grids = {k: np.zeros(floor.shape, dtype=bool) for k in slabs}
    for other in scene.objects.values():
        if (other.object_id in movers or is_opening(other)
                or other.box.z_max <= band[0] or other.box.z_min >= band[1]):
            continue
        touched = slab_range(other.box.z_min, other.box.z_max, height)
        for k in slabs:
            if touched.start - reach <= k < touched.stop + reach:
                # Slabs reached through the vertical margin see the nearest slab.
                nearest = min(max(k, touched.start), touched.stop - 1)
                grids[k] |= _obstacle(other, floor, nearest, height,
                                      inflation + _cell_radius(other), band)
    return grids


def _height_band(scene: Scene, movers: list[int], margin: float,
                 tolerance: float) -> tuple[float, float]:
    """Returns the heights a moving object and what it carries can reach.

    Objects never change height: what lies entirely below or above them (the
    surface they rest on, a shelf they pass under) cannot be hit.

    Args:
        scene: The scene.
        movers: The moving object and everything it carries.
        margin: Required distance to obstacles, in meters.
        tolerance: Interpenetration tolerated by exact placement checks, in meters.

    Returns:
        ``(bottom, top)``, in meters.
    """
    return (min(scene[i].box.z_min for i in movers) - margin + tolerance,
            max(scene[i].box.z_max for i in movers) + margin - tolerance)


def _obstacle(obj: SceneObject, floor: FloorMap, slab: int, height: float,
              inflation: float, band: tuple[float, float]) -> BoolArray:
    """Rasterizes what an object occupies in one slab, inflated.

    Args:
        obj: The obstacle.
        floor: Grid to rasterize onto.
        slab: Slab index, within the object's height range.
        height: Slab height, in meters.
        inflation: Growth of the footprint, in meters.
        band: Heights the moving object can reach; shape cells outside are ignored.

    Returns:
        A (rows, cols) mask.
    """
    if obj.shape is not None:
        return _shape_obstacle(obj, floor, slab, height, inflation, band)
    size = obj.box.size.copy()
    size[:2] = np.maximum(size[:2] + 2.0 * inflation, 0.0)
    return floor.rasterize(OrientedBox(obj.box.center, size, obj.box.yaw).footprint())


def free_of_obstacles(stamps: dict[int, BoolArray], obstacles: dict[int, BoolArray]) -> BoolArray:
    """Finds the positions where no stamp hits an obstacle of its slab.

    Args:
        stamps: Footprint stamps of the moving object, keyed by slab.
        obstacles: Obstacle masks, keyed by slab; all of the same shape.

    Returns:
        A (rows, cols) mask of free center cells.
    """
    shape = next(iter(obstacles.values())).shape
    free = np.ones(shape, dtype=bool)
    for k, stamp in stamps.items():
        grid = obstacles.get(k)
        if grid is None or not grid.any() or not stamp.any():
            continue
        # Correlation with the stamp: convolution with the flipped stamp.
        hits = signal.fftconvolve(grid.astype(np.float64), stamp[::-1, ::-1].astype(np.float64),
                                  mode="same")
        free &= hits < 0.5
    return free
