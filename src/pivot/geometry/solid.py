"""Collision tests between objects given as a box with an optional voxel shape.

Objects without a shape fill their box, and two of them are tested exactly
with :mod:`pivot.geometry.collision`. As soon as one side has a shape, the
test runs on its cells: a chair can then stand under a table top, which a
box-against-box test would forbid. Boxes still serve as a fast first filter,
since cells never leave their box.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import box_distance, box_penetration
from pivot.geometry.shape import VoxelShape
from pivot.geometry.types import BoolArray, FloatArray


def inside_shape(points: FloatArray, box: OrientedBox, shape: VoxelShape,
                  tolerance: float) -> BoolArray:
    """Tests which points lie inside a shape, deeper than a tolerance.

    A point counts when it stays in occupied cells after moving by the
    tolerance along each axis: neighboring objects' grids are not aligned, so
    touching objects overlap by up to half a cell at their boundary.

    Args:
        points: (N, 3) world points.
        box: Box of the shape.
        shape: Occupied cells.
        tolerance: Overlap depth that does not count, in meters.

    Returns:
        (N,) mask.
    """
    local = box.to_local(points)
    inside = shape.contains_cells(np.floor(local / shape.resolution).astype(np.int64))
    for axis in range(3):
        for sign in (-1.0, 1.0):
            shifted = local.copy()
            shifted[:, axis] += sign * tolerance
            inside &= shape.contains_cells(np.floor(shifted / shape.resolution).astype(np.int64))
    result: BoolArray = inside
    return result


def overlap_points(a_box: OrientedBox, a_shape: VoxelShape | None, b_box: OrientedBox,
                   b_shape: VoxelShape | None, tolerance: float = 0.01) -> FloatArray | None:
    """Finds cell centers of one object lying inside the other.

    Args:
        a_box: Box of the first object.
        a_shape: Its shape, or None if it fills its box.
        b_box: Box of the second object.
        b_shape: Its shape, or None.
        tolerance: Overlap depth that does not count, in meters.

    Returns:
        (M, 3) overlapping cell centers, or None when neither object has a shape.
    """
    if a_shape is not None and b_shape is not None:
        # Both ways: either object's cells may reach into the other's.
        a_points, b_points = a_shape.world_centers(a_box), b_shape.world_centers(b_box)
        result: FloatArray = np.vstack([
            a_points[inside_shape(a_points, b_box, b_shape, tolerance)],
            b_points[inside_shape(b_points, a_box, a_shape, tolerance)]])
        return result
    if a_shape is not None:
        points = a_shape.world_centers(a_box)
        result = points[b_box.contains(points, margin=-tolerance)]
        return result
    if b_shape is not None:
        points = b_shape.world_centers(b_box)
        result = points[a_box.contains(points, margin=-tolerance)]
        return result
    return None


def solid_penetration(a_box: OrientedBox, a_shape: VoxelShape | None, b_box: OrientedBox,
                      b_shape: VoxelShape | None, tolerance: float = 0.01) -> float:
    """Measures how deeply two objects overlap.

    With shapes, the depth is the thinnest extent of the overlapping cells
    plus one cell, which approximates the translation needed to separate the
    objects; it never exceeds the overlap of their boxes.

    Args:
        a_box: Box of the first object.
        a_shape: Its shape, or None if it fills its box.
        b_box: Box of the second object.
        b_shape: Its shape, or None.
        tolerance: Overlap depth that does not count, in meters.

    Returns:
        The overlap depth, in meters; 0 if the objects do not overlap.
    """
    box_depth = box_penetration(a_box, b_box)
    if box_depth <= tolerance:
        return 0.0
    points = overlap_points(a_box, a_shape, b_box, b_shape, tolerance)
    if points is None:
        return box_depth
    if len(points) == 0:
        return 0.0
    resolution = (a_shape or b_shape).resolution  # type: ignore[union-attr]
    extent = np.ptp(a_box.to_local(points), axis=0)
    return float(min(box_depth, extent.min() + resolution))


# Obstacles do not move while a motion is planned: their trees are reused.
_TREES: dict[tuple[int, tuple[float, ...]], tuple[VoxelShape, cKDTree]] = {}


def _centers_tree(box: OrientedBox, shape: VoxelShape) -> cKDTree:
    """Returns a KD-tree of a shape's cell centers at a pose, cached.

    Args:
        box: The object's box.
        shape: Its shape.

    Returns:
        The tree.
    """
    key = (id(shape), (*box.center.tolist(), box.yaw))
    cached = _TREES.get(key)
    # The shape is kept in the entry, so its id cannot be reused while cached.
    if cached is None or cached[0] is not shape:
        if len(_TREES) >= 256:
            _TREES.clear()
        cached = (shape, cKDTree(shape.world_centers(box)))
        _TREES[key] = cached
    return cached[1]


def solid_distance(a_box: OrientedBox, a_shape: VoxelShape | None, b_box: OrientedBox,
                   b_shape: VoxelShape | None, horizon: float = 1.0) -> float:
    """Measures the free distance between two objects.

    Args:
        a_box: Box of the first object.
        a_shape: Its shape, or None if it fills its box.
        b_box: Box of the second object.
        b_shape: Its shape, or None.
        horizon: Distances beyond this are not resolved, in meters: the
            result is then only known to be at least the horizon.

    Returns:
        The distance, in meters; 0 if the objects touch or overlap; at least
        ``horizon`` if they are farther apart.
    """
    distance = box_distance(a_box, b_box)
    if distance > horizon or (a_shape is None and b_shape is None):
        return distance
    if a_shape is not None and b_shape is not None:
        # Centers are half a cell inside each object's surface.
        inset = 0.5 * (a_shape.resolution + b_shape.resolution)
        gaps, _ = _centers_tree(b_box, b_shape).query(a_shape.world_centers(a_box),
                                                      distance_upper_bound=horizon + inset)
        return max(min(float(gaps.min()) - inset, horizon), distance)
    shape, shaped_box, box = ((a_shape, a_box, b_box) if a_shape is not None
                              else (b_shape, b_box, a_box))
    assert shape is not None
    local = np.abs(box.to_local(shape.world_centers(shaped_box))) - 0.5 * box.size
    gaps = np.linalg.norm(np.maximum(local, 0.0), axis=1)
    return max(float(gaps.min()) - 0.5 * shape.resolution, distance)
