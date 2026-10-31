"""Assembles a scene graph from a reconstruction and object tracks."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from pivot.geometry.box import OrientedBox, fit_box
from pivot.geometry.floor import FloorMap
from pivot.geometry.plane import apply_transform
from pivot.geometry.types import BoolArray, FloatArray
from pivot.lifting.backproject import scene_points
from pivot.lifting.completion import complete_hidden, drop_fragments
from pivot.lifting.floor import detect_floor, estimate_ceiling, remove_standing, room_orientation
from pivot.lifting.objects import (
    assign_by_votes, contained_tracks, distinct_pairs, duplicate_groups, lift_track,
    split_overlaps,
)
from pivot.lifting.openings import Walls, clean_openings
from pivot.lifting.parts import join_parts
from pivot.lifting.volume import CarveConfig, carve_shapes
from pivot.perception.tracks import Track
from pivot.reconstruction.types import Reconstruction
from pivot.scene.graph import STRUCTURAL_LABELS, Scene, SceneObject, is_opening, is_supported_by


@dataclass(frozen=True)
class MergeConfig:
    """Parameters of merging tracks into objects.

    Attributes:
        duplicate_overlap: Tracks of one label are merged when this fraction of
            the smaller cloud coincides with the larger one or lies in its box.
        part_gap: Separate instances of one label closer than this, and small
            enough together to be one object, are joined as its parts, in meters.
    """

    duplicate_overlap: float = 0.5
    part_gap: float = 0.06


@dataclass(frozen=True)
class BuildConfig:
    """Parameters of scene assembly.

    Attributes:
        voxel: Object fusion voxel size, in meters.
        min_points: Objects with fewer fused voxels are discarded.
        support_tolerance: Gap or interpenetration under which an object counts
            as resting on another one, in meters.
        max_fall: Unsupported objects whose bottom is higher than this are
            taken to be wall-mounted and left in place, in meters.
        floor_band: Points below this height count as floor, in meters.
        floor_resolution: Floor map cell size, in meters.
        merge: Track merging parameters.
        box_trim: Fraction of an object's points ignored at each end of every
            box axis, see :func:`~pivot.geometry.box.fit_box`.
        align_tolerance_deg: Objects within this angle of the walls are
            aligned with them, in degrees.
        carve: Volume carving parameters; None keeps whole boxes.
    """

    voxel: float = 0.03
    min_points: int = 100
    support_tolerance: float = 0.10
    max_fall: float = 1.0
    floor_band: float = 0.05
    floor_resolution: float = 0.05
    merge: MergeConfig = field(default_factory=MergeConfig)
    box_trim: float = 0.02
    align_tolerance_deg: float = 12.0
    carve: CarveConfig | None = field(default_factory=CarveConfig)


@dataclass(frozen=True)
class BuiltScene:
    """A scene and the leveled data it was built from.

    Attributes:
        scene: The scene graph, in the leveled frame (floor at z = 0).
        recon: The reconstruction, transformed to the leveled frame.
        object_points: Fused points of every object, keyed by object id.
    """

    scene: Scene
    recon: Reconstruction
    object_points: dict[int, FloatArray]


def _resting_height(obj: SceneObject, others: list[SceneObject], tolerance: float) -> float:
    """Returns the height an object rests at: the top of its support, or the floor.

    Args:
        obj: The object.
        others: Candidate supports.
        tolerance: Allowed gap or interpenetration with a support's top, in meters.

    Returns:
        The height of the highest support under the object, 0 for the floor.
    """
    # A support's top must be below the object's top, or the object is under it.
    tops = [o.box.z_max for o in others if o.object_id != obj.object_id
            and o.box.z_max < obj.box.z_max
            and is_supported_by(obj.box, o.box, height_tolerance=tolerance)]
    return max(tops, default=0.0)


def settle(objects: list[SceneObject], tolerance: float,
           max_fall: float = 1.0) -> list[SceneObject]:
    """Applies gravity: every movable object rests on a support or on the floor.

    Legs and bottoms are thin or occluded, so fitted boxes often float (a
    table reduced to its top, a chair to its seat). An object that rests on
    no other object is extended down to the floor; one that does is set
    exactly on its support's top. Objects floating higher than ``max_fall``
    (wall shelves, lamp heads) are left as fitted, like structural objects.
    Supports are tested on the fitted boxes, before any change.

    Args:
        objects: Objects with fitted boxes.
        tolerance: Allowed gap or interpenetration with a support's top, in meters.
        max_fall: Largest drop to a support or to the floor, in meters.

    Returns:
        The objects with settled boxes.
    """
    settled = []
    for obj in objects:
        if obj.movable:
            base = _resting_height(obj, objects, tolerance)
            box = obj.box
            if box.z_min - base > max_fall:
                settled.append(obj)
                continue
            size, center = box.size.copy(), box.center.copy()
            size[2] = box.z_max - base
            center[2] = 0.5 * (box.z_max + base)
            obj = replace(obj, box=OrientedBox(center, size, box.yaw))
        settled.append(obj)
    return settled


def build_scene(recon: Reconstruction, tracks: list[Track],
                config: BuildConfig | None = None) -> BuiltScene:
    """Builds a leveled scene graph from posed frames and object tracks.

    Args:
        recon: Posed frames with metric depth, in any world frame.
        tracks: Object tracks whose frame indices refer to ``recon``.
        config: Assembly parameters.

    Returns:
        The scene and its supporting data.
    """
    config = config or BuildConfig()
    points = scene_points(recon)
    _, transform = detect_floor(points, recon.up_direction())
    leveled = recon.transformed(transform)
    points = apply_transform(transform, points)

    room_yaw = room_orientation(points)
    objects, object_points, object_tracks = _lift_objects(leveled, tracks, room_yaw, config)
    objects = clean_openings(objects, object_points, _walls(points, objects, room_yaw, config))

    objects = settle(objects, config.support_tolerance, config.max_fall)
    if config.carve is not None:
        masks = {i: _merged_masks(group) for i, group in object_tracks.items()}
        objects = carve_shapes(leveled, objects, object_points, masks, config.carve)
    objects = complete_hidden(drop_fragments(objects))
    floor = _floor_map(points, objects, config)
    metadata = {"frames": len(recon), "tracks": len(tracks)}
    scene = Scene.from_objects(objects, floor, metadata, estimate_ceiling(points, floor))
    return BuiltScene(scene, leveled, object_points)


def _walls(points: FloatArray, objects: list[SceneObject], room_yaw: float,
           config: BuildConfig) -> Walls:
    """Finds the walls openings are set in, inner walls included.

    Args:
        points: (N, 3) leveled scene points.
        objects: The lifted objects.
        room_yaw: Orientation of the room's walls, in radians.
        config: Assembly parameters.

    Returns:
        The walls.
    """
    floor_xy = points[np.abs(points[:, 2]) < config.floor_band, :2]
    # Thin walls, erased by the floor map's gap filling, are found again from
    # what stands outside the objects (openings excepted: their frames are walls).
    wall_floor = remove_standing(FloorMap.from_points(floor_xy, config.floor_resolution),
                                 points, [o.box for o in objects if not is_opening(o)],
                                 config.floor_band)
    return Walls.from_floor(floor_xy, room_yaw, wall_floor)


def fit_object_box(cloud: FloatArray, room_yaw: float, trim: float = 0.02,
                   align_tolerance_deg: float = 12.0) -> OrientedBox:
    """Fits an object's box, robust to stray points and aligned with the walls.

    Furniture mostly stands parallel to the walls, while the minimum-area
    orientation of a partial, noisy cloud can be off by several degrees.

    Args:
        cloud: (N, 3) object points.
        room_yaw: Wall orientation, see :func:`~pivot.lifting.floor.room_orientation`.
        trim: Fraction of the points ignored at each end of every axis.
        align_tolerance_deg: Boxes within this angle of the walls are aligned
            with them, in degrees.

    Returns:
        The box.
    """
    box = fit_box(cloud, trim=trim)
    quarter = 0.5 * math.pi
    offset = (box.yaw - room_yaw + 0.5 * quarter) % quarter - 0.5 * quarter
    if abs(offset) <= math.radians(align_tolerance_deg):
        box = fit_box(cloud, yaw=box.yaw - offset, trim=trim)
    return box


def _lift_objects(recon: Reconstruction, tracks: list[Track], room_yaw: float, config: BuildConfig
                  ) -> tuple[list[SceneObject], dict[int, FloatArray], dict[int, list[Track]]]:
    """Lifts tracks to objects with fitted, not yet settled, boxes.

    Each object takes the id and label of the first track of its group.
    Small objects and flat patches of floor are discarded.

    Args:
        recon: The leveled reconstruction.
        tracks: Object tracks.
        room_yaw: Wall orientation, in radians.
        config: Assembly parameters.

    Returns:
        The objects, and their fused point clouds and source tracks keyed by object id.
    """
    objects: list[SceneObject] = []
    object_points: dict[int, FloatArray] = {}
    object_tracks: dict[int, list[Track]] = {}

    def fit(cloud: FloatArray) -> OrientedBox:
        """Fits a box with the configured trimming and wall alignment.

        Args:
            cloud: (N, 3) object points.

        Returns:
            The box.
        """
        return fit_object_box(cloud, room_yaw, config.box_trim, config.align_tolerance_deg)

    for group, cloud in _merged_clouds(recon, tracks, fit, config):
        if len(cloud) < config.min_points:
            continue
        box = fit(cloud)
        if box.z_max < config.floor_band:
            continue  # A patch of floor segmented as an object.
        first = group[0]
        movable = first.label.lower() not in STRUCTURAL_LABELS
        score = max(t.score for t in group)
        objects.append(SceneObject(first.track_id, first.label, box, movable, score))
        object_points[first.track_id] = cloud
        object_tracks[first.track_id] = group
    return objects, object_points, object_tracks


def _merged_masks(tracks: list[Track]) -> dict[int, BoolArray]:
    """Unites the masks of tracks describing one object, frame by frame.

    Args:
        tracks: Tracks of one object.

    Returns:
        Masks keyed by frame.
    """
    merged: dict[int, BoolArray] = {}
    for track in tracks:
        for frame, mask in track.masks.items():
            merged[frame] = merged[frame] | mask if frame in merged else mask
    return merged


def _merged_clouds(recon: Reconstruction, tracks: list[Track],
                   fit: Callable[[FloatArray], OrientedBox],
                   config: BuildConfig) -> list[tuple[list[Track], FloatArray]]:
    """Lifts tracks and merges those of one label that describe the same object.

    Args:
        recon: The leveled reconstruction.
        tracks: Object tracks.
        fit: Box fitting function.
        config: Assembly parameters.

    Returns:
        Each group of tracks with its fused point cloud, in track order.
    """
    clouds = [lift_track(recon, track, voxel=config.voxel) for track in tracks]
    boxes = [fit(c) if len(c) else None for c in clouds]
    # A part segmented under another noun is already in the whole's points.
    parts = contained_tracks(tracks)
    merged = []
    for label in dict.fromkeys(t.label for t in tracks):
        indices = [i for i, t in enumerate(tracks) if t.label == label and i not in parts]
        if not indices:
            continue
        groups = [[indices[i] for i in group] for group in duplicate_groups(
            [clouds[i] for i in indices], 1.5 * config.voxel, config.merge.duplicate_overlap,
            [boxes[i] for i in indices], distinct=distinct_pairs([tracks[i] for i in indices]))]
        # Members are voxel centers on the same grid: duplicates coincide.
        group_clouds = [np.unique(np.vstack([clouds[i] for i in g]), axis=0) for g in groups]
        objects = [sorted(i for p in parts for i in groups[p])
                   for parts in join_parts(group_clouds, label, fit, config.merge.part_gap)]
        # Tracks are sparse: support is judged over a few voxels around each point.
        owned = split_overlaps(assign_by_votes(
            [[clouds[i] for i in members] for members in objects], 3.0 * config.voxel), fit)
        merged += [([tracks[i] for i in members], cloud)
                   for members, cloud in zip(objects, owned)]
    merged.sort(key=lambda item: item[0][0].track_id)
    return merged


def _floor_map(points: FloatArray, objects: list[SceneObject], config: BuildConfig) -> FloorMap:
    """Builds the walkable floor map of a leveled scene.

    Floor hidden under furniture is still floor: the footprints of movable
    objects standing on it are added so that they can be moved elsewhere and
    other objects can take their place. Walls are taken off again after the
    floor's gaps are filled (see :func:`~pivot.lifting.floor.remove_standing`).

    Args:
        points: (N, 3) leveled scene points.
        objects: Objects of the scene.
        config: Assembly parameters.

    Returns:
        The floor map.
    """
    floor_xy = points[np.abs(points[:, 2]) < config.floor_band, :2]
    hidden = [_fill_polygon(o.box.footprint(), config.floor_resolution / 2)
              for o in objects if o.movable and o.box.z_min < config.floor_band]
    floor = FloorMap.from_points(np.vstack([floor_xy, *hidden]), config.floor_resolution)
    # Gap filling erases thin walls: take off what stands outside the objects.
    return remove_standing(floor, points, [o.box for o in objects if not is_opening(o)],
                           config.floor_band)


def _fill_polygon(polygon: FloatArray, spacing: float) -> FloatArray:
    """Samples a regular grid of points inside a rectangle footprint.

    Args:
        polygon: (4, 2) rectangle corners in order.
        spacing: Grid spacing, in meters.

    Returns:
        (N, 2) points covering the rectangle.
    """
    side_u, side_v = polygon[1] - polygon[0], polygon[3] - polygon[0]
    steps_u = max(2, int(np.linalg.norm(side_u) / spacing) + 1)
    steps_v = max(2, int(np.linalg.norm(side_v) / spacing) + 1)
    u, v = np.meshgrid(np.linspace(0, 1, steps_u), np.linspace(0, 1, steps_v))
    result: FloatArray = polygon[0] + u.reshape(-1, 1) * side_u + v.reshape(-1, 1) * side_v
    return result
