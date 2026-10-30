"""Completing objects partly hidden behind or under others, from size priors.

The camera only sees what is not occluded: a chair pushed under a desk shows
little more than its back. Moved elsewhere, such an object would carry only
its visible part. When an object is smaller than any object of its label can
be and touches another object, its hidden part is assumed to extend towards
that object: its box grows to the minimum size of its label on that side.
The new part continues the visible one at the same heights: the visible
cells next to the hidden side are extruded into it (a chair's seat goes on
under the desk, its backrest does not), and only where no other object is.

A much smaller fragment stuck to an object of the same label is not a hidden
object but a misplaced view of that one (the top of a backrest seen from far
away, a few tens of centimeters off): it is dropped rather than completed.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from scipy.spatial import cKDTree

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.geometry.collision import box_distance
from pivot.geometry.shape import VoxelShape
from pivot.geometry.types import BoolArray, FloatArray
from pivot.scene.graph import SceneObject, is_opening

# Smallest footprint of one object of a label, (long side, short side) in meters.
MIN_FOOTPRINT = {"chair": (0.5, 0.5), "office chair": (0.55, 0.55), "armchair": (0.6, 0.6),
                 "stool": (0.3, 0.3), "trash can": (0.25, 0.25)}


def _occluder(obj: SceneObject, others: list[SceneObject],
              max_gap: float) -> SceneObject | None:
    """Finds the object most likely to hide part of another: the nearest touching one.

    Args:
        obj: The object.
        others: Candidate occluders.
        max_gap: Largest distance to an occluding object, in meters.

    Returns:
        The occluder, or None if nothing touches the object.
    """
    near = [o for o in others if box_distance(obj.box, o.box) <= max_gap]
    return min(near, key=lambda o: box_distance(obj.box, o.box)) if near else None


def _hidden_side(obj: SceneObject, occluder: SceneObject, axis: int) -> float:
    """Finds on which side of an axis an object's hidden part lies.

    Args:
        obj: The object.
        occluder: The object hiding part of it.
        axis: Box axis (0 or 1).

    Returns:
        +1 or -1 along the axis, towards the occluder.
    """
    offset = obj.box.to_local(occluder.box.center[None])[0, axis]
    return 1.0 if offset >= 0.0 else -1.0


def _grown(box: OrientedBox, axis: int, size: float, side: float) -> OrientedBox:
    """Grows a box along one of its axes, keeping the opposite face in place.

    Args:
        box: The box.
        axis: Box axis (0 or 1).
        size: New size along the axis, in meters.
        side: +1 or -1: the direction the box grows in.

    Returns:
        The grown box.
    """
    direction = np.zeros(3)
    direction[axis] = side
    shift = 0.5 * (size - box.size[axis]) * direction
    center = box.center.copy()
    center[:2] += rotation_2d(box.yaw) @ shift[:2]
    new_size = box.size.copy()
    new_size[axis] = size
    return OrientedBox(center, new_size, box.yaw)


def _extruded(old_local: FloatArray, old_box: OrientedBox, box: OrientedBox,
              local: FloatArray, resolution: float, band: int = 3) -> BoolArray:
    """Tests which new cells continue the visible part, at the same heights.

    A new cell is kept when the visible part has a cell at its height near
    the point of the original box closest to it: the visible boundary is
    extruded outwards.

    Args:
        old_local: (M, 3) centers of the visible cells, in the grown box's frame.
        old_box: The original box.
        box: The grown box, with the same yaw.
        local: (N, 3) centers of the new cells, in the grown box's frame.
        resolution: Cell size, in meters.
        band: Horizontal search radius, in cells.

    Returns:
        (N,) mask of the kept cells.
    """
    if len(old_local) == 0 or len(local) == 0:
        return np.zeros(len(local), dtype=bool)
    offset = box.to_local(old_box.center[None])[0]
    half = 0.5 * old_box.size
    nearest = np.clip(local, offset - half, offset + half)
    nearest[:, 2] = local[:, 2]
    # Heights must match exactly: the vertical axis is stretched.
    stretch = np.array([1.0, 1.0, 1e3])
    distance, _ = cKDTree(old_local * stretch).query(nearest * stretch)
    return np.asarray(distance <= band * resolution)


def _fill(obj: SceneObject, box: OrientedBox, others: list[SceneObject],
          top: float) -> VoxelShape | None:
    """Rebuilds an object's shape in its grown box, extending its visible part.

    Args:
        obj: The object, with its original box and shape.
        box: Its grown box.
        others: Objects whose occupied space the new part must avoid.
        top: The new part stays below this height: higher, the camera would
            have seen it.

    Returns:
        The shape in the grown box's frame, or None if the object had none.
    """
    if obj.shape is None:
        return None
    resolution = obj.shape.resolution
    old = obj.shape.world_centers(obj.box)
    half = np.ceil(0.5 * box.size / resolution).astype(np.int64)
    axes = [np.arange(-h, h) for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    candidate = VoxelShape(resolution, grid)
    centers = candidate.world_centers(box)
    # One cell below the top: the occluder's top cells may not reach its box top.
    new = ~obj.box.contains(centers) & box.contains(centers) & (centers[:, 2] <= top - resolution)
    new[new] = _extruded(box.to_local(old), obj.box, box, box.to_local(centers[new]), resolution)
    # A cell of margin around the others: cells filled between theirs would
    # interlock the two volumes, and the object could no longer slide out.
    margin = 1.5 * resolution
    for other in others:
        if other.shape is not None:
            near, _ = cKDTree(other.shape.world_centers(other.box)).query(
                centers[new], distance_upper_bound=margin)
            new[new] = ~np.isfinite(near)
        else:
            new &= ~other.box.contains(centers)
    return VoxelShape.from_world_points(box, np.vstack([old, centers[new]]), resolution)


def drop_fragments(objects: list[SceneObject], max_gap: float = 0.1,
                   max_fraction: float = 0.5) -> list[SceneObject]:
    """Drops fragments that are misplaced views of a neighbor of the same label.

    Args:
        objects: Objects with settled boxes.
        max_gap: Largest distance to the neighbor, in meters.
        max_fraction: A fragment's footprint is smaller than this fraction of
            the minimum footprint of its label.

    Returns:
        The objects, without the fragments.
    """
    return [o for o in objects if not _is_fragment(o, objects, max_gap, max_fraction)]


def _is_fragment(obj: SceneObject, objects: list[SceneObject], max_gap: float,
                 max_fraction: float) -> bool:
    """Tests whether an object is a small misplaced view of a neighbor of its label.

    Args:
        obj: The object.
        objects: All objects.
        max_gap: Largest distance to the neighbor, in meters.
        max_fraction: Footprint fraction of the label's minimum under which
            the object is a fragment.

    Returns:
        True if the object should be dropped.
    """
    minimum = MIN_FOOTPRINT.get(obj.label.lower())
    area = obj.box.size[0] * obj.box.size[1]
    if minimum is None or area >= max_fraction * minimum[0] * minimum[1]:
        return False
    return any(o.object_id != obj.object_id and o.label == obj.label
               and o.box.size[0] * o.box.size[1] > area
               and box_distance(obj.box, o.box) <= max_gap for o in objects)


def complete_hidden(objects: list[SceneObject], max_gap: float = 0.1) -> list[SceneObject]:
    """Grows objects too small for their label towards what hides them.

    Args:
        objects: Objects with settled boxes and, possibly, shapes.
        max_gap: Largest distance to an occluding object, in meters.

    Returns:
        The objects, completed where needed.
    """
    result = []
    for obj in objects:
        minimum = MIN_FOOTPRINT.get(obj.label.lower())
        if minimum is None or not obj.movable:
            result.append(obj)
            continue
        others = [o for o in objects if o.object_id != obj.object_id and not is_opening(o)]
        occluder = _occluder(obj, others, max_gap)
        box = obj.box
        for axis, size in enumerate(minimum):
            if occluder is not None and box.size[axis] < size:
                box = _grown(box, axis, size, _hidden_side(obj, occluder, axis))
        if occluder is None or box is obj.box:
            result.append(obj)
        else:
            shape = _fill(obj, box, others, occluder.box.z_max)
            result.append(dataclasses.replace(obj, box=box, shape=shape))
    return result
