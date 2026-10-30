"""Carving the actual volume of objects out of their boxes with depth maps.

Cameras only see surfaces, so the inside of an object is never observed. A
cell of space is known to be free only when some camera saw *through* it,
that is saw a surface behind it. Each candidate cell of an object's box is
projected into every depth map and classified:

* free: seen through in several views (the space under a table);
* surface: lies on an observed surface, its own or another one (floor, wall);
* unknown: never seen through nor on a surface (the inside of a closed box).

An object occupies the cells holding its own points plus the unknown cells
inside its box: unknown space is kept as occupied, which is conservative.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import box_distance
from pivot.geometry.shape import VoxelShape
from pivot.geometry.solid import inside_shape
from pivot.geometry.types import BoolArray, FloatArray, IntArray
from pivot.reconstruction.types import Reconstruction
from pivot.scene.graph import SceneObject


@dataclass(frozen=True)
class CarveConfig:
    """Parameters of volume carving.

    Attributes:
        resolution: Cell side length, in meters.
        frame_stride: Use one frame out of ``frame_stride``.
        surface_band: Depth difference under which a cell lies on the observed
            surface, at zero depth, in meters.
        surface_band_slope: Growth of that band with depth, absorbing depth
            noise, in meters per meter.
        min_free_views: Views that must see through a cell to call it free.
        min_mask_views: Views that must see a cell on a surface inside the
            object's mask to make it part of the object.
        max_depth: Depths beyond this are ignored, in meters.
    """

    resolution: float = 0.04
    frame_stride: int = 2
    surface_band: float = 0.04
    surface_band_slope: float = 0.01
    min_free_views: int = 2
    min_mask_views: int = 2
    max_depth: float = 6.0


@dataclass(frozen=True)
class Observations:
    """What the cameras saw of a set of points, as per-point view counts.

    Attributes:
        free: Views that saw through the point.
        surface: Views that saw the point on the observed surface.
        masked: Views where the point lies on the surface inside its owner's
            mask: thin parts (legs, frames) that the eroded point cloud loses.
        mask_views: Surface views in frames where the owner has a mask.
    """

    free: IntArray
    surface: IntArray
    masked: IntArray
    mask_views: IntArray

    def subset(self, selection: BoolArray) -> Observations:
        """Returns the counts of a subset of the points.

        Args:
            selection: (N,) mask of the points to keep.

        Returns:
            Their counts.
        """
        return Observations(self.free[selection], self.surface[selection],
                            self.masked[selection], self.mask_views[selection])


@dataclass(frozen=True)
class _View:
    """Projection of points into one depth map.

    Attributes:
        u: (N,) column coordinates, in depth pixels.
        v: (N,) row coordinates, in depth pixels.
        free: (N,) points seen through.
        surface: (N,) points on the observed surface.
    """

    u: FloatArray
    v: FloatArray
    free: BoolArray
    surface: BoolArray


def _project(recon: Reconstruction, frame: int, points: FloatArray, config: CarveConfig) -> _View:
    """Projects points into one depth map and compares them with it.

    Args:
        recon: Posed depth maps.
        frame: Frame index.
        points: (N, 3) world points.
        config: Carving parameters.

    Returns:
        The projection.
    """
    world_to_cam = np.linalg.inv(recon.cam_to_world[frame])
    cam = points @ world_to_cam[:3, :3].T + world_to_cam[:3, 3]
    in_front = cam[:, 2] > 0.1
    z = np.where(in_front, cam[:, 2], 1.0)
    pixels = cam[:, :2] / z[:, None] @ recon.depth_intrinsics(frame)[:2, :2].T \
        + recon.depth_intrinsics(frame)[:2, 2]
    # Pixel (row, col) covers [col, col + 1) x [row, row + 1) (OpenCV centers at + 0.5).
    index = np.floor(pixels).astype(np.int64)
    seen = in_front & np.all((index >= 0) & (index < np.array(recon.depth_size)), axis=1)
    depth = np.zeros(len(points))
    depth[seen] = recon.depths[frame][index[seen, 1], index[seen, 0]]
    seen &= (depth > 0) & (depth < config.max_depth)
    band = config.surface_band + config.surface_band_slope * depth
    return _View(pixels[:, 0], pixels[:, 1], seen & (z < depth - band),
                 seen & (np.abs(z - depth) <= band))


def observation_counts(recon: Reconstruction, points: FloatArray, config: CarveConfig,
                       owners: IntArray | None = None,
                       masks: list[dict[int, BoolArray]] | None = None) -> Observations:
    """Counts, for each point, the views that see through it and those that see it on a surface.

    Args:
        recon: Posed depth maps.
        points: (N, 3) world points.
        config: Carving parameters.
        owners: Optional (N,) index of the object each point belongs to.
        masks: Masks of each object keyed by frame, at any resolution; used
            with ``owners``.

    Returns:
        The view counts.
    """
    seen = Observations(*(np.zeros(len(points), dtype=np.int64) for _ in range(4)))
    members = [np.flatnonzero(owners == k) for k in range(len(masks))] \
        if owners is not None and masks is not None else []
    for frame in range(0, len(recon), config.frame_stride):
        view = _project(recon, frame, points, config)
        seen.free[:] += view.free
        seen.surface[:] += view.surface
        for index, member_ids in enumerate(members):
            mask = masks[index].get(frame) if masks is not None else None
            if mask is not None:
                _count_mask(seen, view, member_ids[view.surface[member_ids]], mask,
                            recon.depth_size)
    return seen


def _count_mask(seen: Observations, view: _View, ids: IntArray, mask: BoolArray,
                depth_size: tuple[int, int]) -> None:
    """Counts the surface views of points that fall inside their owner's mask.

    Args:
        seen: Counts to update in place.
        view: Projection of all points into the frame.
        ids: Points of one object that lie on the observed surface.
        mask: That object's mask in the frame, at any resolution.
        depth_size: Depth map size ``(width, height)``.
    """
    # Masks may be finer than depth: sample them at the exact projection.
    cols = np.clip((view.u[ids] * mask.shape[1] / depth_size[0]).astype(np.int64),
                   0, mask.shape[1] - 1)
    rows = np.clip((view.v[ids] * mask.shape[0] / depth_size[1]).astype(np.int64),
                   0, mask.shape[0] - 1)
    seen.mask_views[ids] += 1
    seen.masked[ids] += mask[rows, cols]


def _candidate_cells(box: OrientedBox, resolution: float) -> IntArray:
    """Lists the cells of a box's grid covering the box plus one cell around it.

    Args:
        box: The box.
        resolution: Cell side length, in meters.

    Returns:
        (M, 3) cell indices in the box frame.
    """
    lo = np.floor(-0.5 * box.size / resolution).astype(np.int64) - 1
    hi = np.ceil(0.5 * box.size / resolution).astype(np.int64) + 1
    axes = [np.arange(a, b) for a, b in zip(lo, hi)]
    result: IntArray = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return result


def _owned(index: int, centers: FloatArray, objects: list[SceneObject],
           trees: list[cKDTree | None], resolution: float) -> FloatArray:
    """Returns, for each cell, the distance to its object's points, or inf if another owns it.

    Where boxes overlap (a chair tucked under a table), a cell goes to the
    object with the nearest points.

    Args:
        index: Index of the object in ``objects``.
        centers: (M, 3) world centers of its candidate cells.
        objects: All objects.
        trees: KD-trees of every object's points, None for empty clouds.
        resolution: Cell side length, in meters.

    Returns:
        (M,) distances to the object's own points, inf for cells owned by another object.
    """
    own_tree = trees[index]
    distance = np.full(len(centers), np.inf)
    if own_tree is not None:
        distance, _ = own_tree.query(centers)
    for other_index, other in enumerate(objects):
        tree = trees[other_index]
        if other_index == index or tree is None:
            continue
        shared = other.box.contains(centers, margin=resolution)
        if not shared.any():
            continue
        other_distance, _ = tree.query(centers[shared])
        lost = np.zeros(len(centers), dtype=bool)
        lost[shared] = other_distance < distance[shared]
        distance[lost] = np.inf
    result: FloatArray = distance
    return result


def carve_shapes(recon: Reconstruction, objects: list[SceneObject],
                 clouds: dict[int, FloatArray], masks: dict[int, dict[int, BoolArray]],
                 config: CarveConfig | None = None) -> list[SceneObject]:
    """Gives every object the volume it actually occupies inside its box.

    Args:
        recon: Posed depth maps, in the frame of the boxes.
        objects: Objects with their boxes.
        clouds: Fused points of each object, keyed by object id.
        masks: Masks of each object keyed by object id, then by frame.
        config: Carving parameters.

    Returns:
        The objects with their ``shape`` set; objects without points keep none.
    """
    config = config or CarveConfig()
    if not objects:
        return []
    candidates = [VoxelShape(config.resolution, _candidate_cells(o.box, config.resolution))
                  for o in objects]
    centers = [c.world_centers(o.box) for o, c in zip(objects, candidates)]
    # Index of the object each candidate cell belongs to.
    owners = np.repeat(np.arange(len(objects)), [len(c) for c in centers])
    seen = observation_counts(recon, np.vstack(centers), config, owners,
                              [masks.get(o.object_id, {}) for o in objects])
    trees = [cKDTree(clouds[o.object_id]) if len(clouds.get(o.object_id, [])) else None
             for o in objects]

    result = []
    for index, obj in enumerate(objects):
        if trees[index] is None:
            result.append(obj)
            continue
        occupied = _occupied(obj.box, centers[index], seen.subset(owners == index),
                             _owned(index, centers[index], objects, trees, config.resolution),
                             config)
        result.append(replace(obj, shape=VoxelShape(config.resolution,
                                                    candidates[index].cells[occupied])))
    return separate_shapes(result, trees)


def _contested(boxes: dict[int, OrientedBox], shapes: dict[int, VoxelShape], i: int, j: int,
               tolerance: float) -> BoolArray:
    """Finds the cells of one object lying inside another's shape.

    Args:
        boxes: Boxes of the objects.
        shapes: Shapes of the objects.
        i: Object whose cells are tested.
        j: The other object.
        tolerance: Overlap depth that does not count, in meters.

    Returns:
        A mask over the cells of object ``i``.
    """
    if box_distance(boxes[i], boxes[j]) > 0:
        return np.zeros(len(shapes[i]), dtype=bool)
    return inside_shape(shapes[i].world_centers(boxes[i]), boxes[j], shapes[j], tolerance)


def _nearest_owner(boxes: dict[int, OrientedBox], shapes: dict[int, VoxelShape],
                   trees: dict[int, cKDTree], index: int, tolerance: float) -> VoxelShape:
    """Drops the contested cells of an object that another object's points are nearer to.

    Args:
        boxes: Boxes of the objects.
        shapes: Shapes of the objects.
        trees: KD-trees of the points of the objects being separated.
        index: The object.
        tolerance: Overlap depth that does not count, in meters.

    Returns:
        Its reduced shape.
    """
    shape = shapes[index]
    keep = np.ones(len(shape), dtype=bool)
    for other in trees:
        contested = _contested(boxes, shapes, index, other, tolerance) if other != index \
            else np.zeros(len(shape), dtype=bool)
        if contested.any():
            centers = shape.world_centers(boxes[index])[contested]
            own, _ = trees[index].query(centers)
            theirs, _ = trees[other].query(centers)
            keep[np.flatnonzero(contested)[theirs < own]] = False
    return VoxelShape(shape.resolution, shape.cells[keep])


def separate_shapes(objects: list[SceneObject], trees: list[cKDTree | None],
              tolerance: float = 0.01) -> list[SceneObject]:
    """Removes the overlaps left between neighboring objects' shapes.

    Each object's cells lie on its own grid, so the boundary cells of two
    touching objects can overlap. First, a contested cell stays with the
    object whose points are nearest to its center. Cells that both objects
    still claim are then dropped from both, leaving at most one cell of gap
    between objects that touch anyway.

    Args:
        objects: Objects with their shapes.
        trees: KD-trees of each object's points, None for empty clouds.
        tolerance: Overlap depth that does not count, see
            :func:`~pivot.geometry.solid.inside_shape`.

    Returns:
        The objects with disjoint shapes.
    """
    boxes = {i: o.box for i, o in enumerate(objects)}
    shapes = {i: o.shape for i, o in enumerate(objects) if o.shape is not None}
    owners = {i: tree for i, tree in enumerate(trees) if tree is not None and i in shapes}
    nearest = {i: _nearest_owner(boxes, shapes, owners, i, tolerance) for i in owners}
    disjoint = {}
    for i, shape in nearest.items():
        keep = np.ones(len(shape), dtype=bool)
        for j in nearest:
            if j != i:
                keep &= ~_contested(boxes, nearest, i, j, tolerance)
        disjoint[i] = VoxelShape(shape.resolution, shape.cells[keep])
    return [replace(o, shape=disjoint[i]) if i in disjoint else o for i, o in enumerate(objects)]


def _occupied(box: OrientedBox, centers: FloatArray, seen: Observations,
              own_distance: FloatArray, config: CarveConfig) -> BoolArray:
    """Selects the cells an object occupies.

    A cell is the object's surface when it holds the object's points, or when
    most views that see it on a surface see it inside the object's mask (thin
    legs, lost by mask erosion and depth edge removal). Unknown space inside
    the box is kept as occupied.

    Args:
        box: The object's box.
        centers: (M, 3) world centers of its candidate cells.
        seen: View counts of its candidate cells.
        own_distance: (M,) distance to the object's points, inf where another
            object owns the cell.
        config: Carving parameters.

    Returns:
        (M,) mask of occupied cells.
    """
    owned = np.isfinite(own_distance)
    seen_through = (seen.free >= config.min_free_views) & (seen.free >= seen.surface)
    in_mask = (seen.masked >= config.min_mask_views) & (2 * seen.masked >= seen.mask_views)
    own_surface = (own_distance <= 0.75 * config.resolution) | (in_mask & ~seen_through & owned)
    # Unknown space inside the box: never seen through, and not some other
    # surface (floor, wall) either.
    hidden = ~seen_through & (seen.surface == 0) & box.contains(centers) & owned
    result: BoolArray = own_surface | hidden
    return result
