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
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.geometry.collision import box_distance
from pivot.geometry.floor import FloorMap
from pivot.geometry.shape import VoxelShape
from pivot.geometry.types import BoolArray, FloatArray, UInt8Array
from pivot.scene.graph import Scene, SceneObject, is_flat, is_opening

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


# A rug continues under furniture by at most this much, in meters.
_MAX_FLAT_EXTENSION = 2.0
# Share of a rug's edge that must run along hidden floor for it to continue there:
# where the floor beyond an edge is seen without the rug, the rug ends there.
_HIDDEN_EDGE_SHARE = 0.8
# Spacing of the points tested along an edge, and how far beyond it, in meters.
_EDGE_STEP = 0.02


def _hidden(points: FloatArray, occluders: list[OrientedBox]) -> BoolArray:
    """Tests which floor points are hidden under furniture.

    Args:
        points: (N, 2) floor points.
        occluders: Boxes of the furniture standing on the floor.

    Returns:
        (N,) True for points inside an occluder's footprint.
    """
    hidden = np.zeros(len(points), dtype=bool)
    for box in occluders:
        local = (points - box.center[:2]) @ rotation_2d(box.yaw)
        hidden |= np.all(np.abs(local) <= 0.5 * box.size[:2], axis=1)
    return hidden


def _flat_extent(box: OrientedBox, occluders: list[OrientedBox], floor: FloorMap | None,
                 axis: int, side: float) -> float:
    """Measures how far a rug may continue under furniture beyond one of its edges.

    Its edge must run along hidden floor for most of its length (floor seen
    bare beyond an edge means the rug ends there); the rug then goes on as long
    as the floor beyond stays hidden, up to the far side of what hides it, and
    never off the floor.

    Args:
        box: The rug's box.
        occluders: Boxes of the furniture standing on the floor.
        floor: The floor map, if known.
        axis: 0 or 1, the box axis the edge is perpendicular to.
        side: +1 or -1, which edge.

    Returns:
        The extension, in meters (0 if the edge is seen whole).
    """
    half = 0.5 * box.size[:2]
    other = 1 - axis
    along = np.arange(-half[other], half[other] + 1e-9, _EDGE_STEP)
    rotation = rotation_2d(box.yaw)
    extension = 0.0
    while extension < _MAX_FLAT_EXTENSION:
        local = np.zeros((len(along), 2))
        local[:, other] = along
        local[:, axis] = side * (half[axis] + extension + 0.5 * _EDGE_STEP)
        points = local @ rotation.T + box.center[:2]
        if _hidden(points, occluders).mean() < _HIDDEN_EDGE_SHARE:
            break
        if floor is not None and not all(floor.is_floor(p) for p in points[::5]):
            break
        extension += _EDGE_STEP
    return extension


def complete_flat(objects: list[SceneObject], floor: FloorMap | None = None
                  ) -> list[SceneObject]:
    """Completes rugs partly hidden under furniture, as rectangles.

    A rug is seen only where nothing stands on it: under a bed, the camera
    sees the bed. As a rug is a rectangle, each edge that runs along hidden
    floor is pushed on under what hides it, up to its far side; the edges seen
    whole stay. Taking the bed out then shows the whole rug, not a strip.

    Args:
        objects: Objects with settled boxes.
        floor: The floor map, which the rugs never leave; unchecked if None.

    Returns:
        The objects, rugs completed where needed (as plain boxes).
    """
    occluders = [o.box for o in objects
                 if not (is_flat(o) or is_opening(o)) and o.box.z_min < 0.1]
    result = []
    for obj in objects:
        if not is_flat(obj):
            result.append(obj)
            continue
        box = obj.box
        for axis in (0, 1):
            for side in (1.0, -1.0):
                extension = _flat_extent(box, occluders, floor, axis, side)
                if extension > 0.05:
                    box = _grown(box, axis, float(box.size[axis]) + extension, side)
        result.append(obj if box is obj.box else dataclasses.replace(obj, box=box, shape=None))
    return result


def _floor_grid(box: OrientedBox, spacing: float) -> FloatArray:
    """Spreads points evenly over a box's footprint.

    Args:
        box: The box.
        spacing: Distance between the points, in meters.

    Returns:
        (N, 2) floor points.
    """
    half = 0.5 * box.size[:2]
    grid = np.stack(np.meshgrid(np.arange(-half[0], half[0], spacing) + 0.5 * spacing,
                                np.arange(-half[1], half[1], spacing) + 0.5 * spacing),
                    axis=-1).reshape(-1, 2)
    result: FloatArray = grid @ rotation_2d(box.yaw).T + box.center[:2]
    return result


def hidden_flat_points(before: SceneObject, after: SceneObject, objects: list[SceneObject],
                       points: FloatArray, colors: UInt8Array, spacing: float = 0.03
                       ) -> tuple[FloatArray, UInt8Array]:
    """Makes points for the part of a rug that :func:`complete_flat` added.

    They lie on the rug, in its usual color, only under the furniture that hid
    it: they show once that furniture is moved or taken out.

    Args:
        before: The rug as seen.
        after: The rug completed.
        objects: The objects of the scene, to find what hides the rug.
        points: (N, 3) points of the cloud, to take the rug's color from.
        colors: (N, 3) their colors.
        spacing: Distance between the new points, in meters.

    Returns:
        The (M, 3) new points and their (M, 3) colors.
    """
    xy = _floor_grid(after.box, spacing)
    seen = before.box.to_local(np.column_stack([xy, np.full(len(xy), before.box.center[2])]))
    outside = np.any(np.abs(seen[:, :2]) > 0.5 * before.box.size[:2], axis=1)
    occluders = [o.box for o in objects if not (is_flat(o) or is_opening(o))
                 and o.box.z_min < 0.1]
    keep = outside & _hidden(xy, occluders)
    inside = np.all(np.abs(before.box.to_local(points)) <= 0.5 * before.box.size, axis=1)
    color = np.median(colors[inside], axis=0) if inside.any() else np.full(3, 128.0)
    new = np.column_stack([xy[keep], np.full(int(keep.sum()), after.box.z_max - 0.01)])
    return new, np.tile(color, (len(new), 1)).astype(np.uint8)


def complete_rugs(scene: Scene, points: FloatArray, colors: UInt8Array
                  ) -> tuple[Scene, FloatArray, UInt8Array]:
    """Completes the rugs of a scene and adds points where they were hidden.

    Applied once more to a completed scene, it changes nothing.

    Args:
        scene: The scene.
        points: (N, 3) points of its cloud.
        colors: (N, 3) their colors.

    Returns:
        The scene with its rugs completed, and the cloud with their hidden parts.
    """
    objects = list(scene.objects.values())
    completed = complete_flat(objects, scene.floor)
    new_points, new_colors = [points], [colors]
    for before, after in zip(objects, completed):
        if after is not before:
            extra, extra_colors = hidden_flat_points(before, after, objects, points, colors)
            new_points.append(extra.astype(points.dtype))
            new_colors.append(extra_colors)
    scene = dataclasses.replace(scene, objects={o.object_id: o for o in completed})
    return scene, np.concatenate(new_points), np.concatenate(new_colors)


WINDOW_WALL = 0.15
"""Thickness of the wall a window is closed with, in meters: at least three cells."""


def close_windows(scene: Scene) -> Scene:
    """Takes the floor under windows out of the floor map: nothing passes through one.

    Glass gives few depth points, so the floor seen through a window can be
    mapped as floor, joined to the room's: the window would then be a way
    out. Each window's box, grown to a wall's thickness and a little along
    the wall to meet it, is closed. Applied once more, it changes nothing.

    Args:
        scene: The scene.

    Returns:
        The scene, its floor map closed under its windows.
    """
    floor = scene.floor
    if floor is None:
        return scene
    walkable = floor.floor.copy()
    for obj in scene.objects.values():
        if obj.label.lower() != "window":
            continue
        size = obj.box.size.copy()
        thin = int(np.argmin(size[:2]))
        size[thin] = max(float(size[thin]), WINDOW_WALL)
        size[1 - thin] += WINDOW_WALL
        walkable &= ~floor.rasterize(OrientedBox(obj.box.center, size, obj.box.yaw).footprint())
    if np.array_equal(walkable, floor.floor):
        return scene
    return dataclasses.replace(scene, floor=FloorMap(floor.origin, floor.resolution, walkable))


def prepared_scene(scene_dir: Path) -> tuple[Scene, FloatArray, UInt8Array]:
    """Loads a scene written by :mod:`pivot.build`, completed as the demo shows it.

    Scenes built before windows were closed and rugs completed get it here (a
    no-op for the others), and the floor map learns where the walls stand.

    Args:
        scene_dir: Directory with ``scene.json`` and ``cloud.npz``.

    Returns:
        The scene and its cloud (points and colors).
    """
    cloud = np.load(scene_dir / "cloud.npz")
    scene, points, colors = complete_rugs(close_windows(Scene.load(scene_dir / "scene.json")),
                                          cloud["points"], cloud["colors"])
    if scene.floor is not None:
        # Where the walls stand: paintings hang on them.
        scene = dataclasses.replace(scene, floor=scene.floor.with_walls(points))
    return scene, points, colors
