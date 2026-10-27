"""Assembles a scene graph from a reconstruction and object tracks."""

from dataclasses import dataclass

import numpy as np

from pivot.geometry.box import OrientedBox, fit_box
from pivot.geometry.floor import FloorMap
from pivot.geometry.plane import apply_transform
from pivot.geometry.types import FloatArray
from pivot.lifting.backproject import scene_points
from pivot.lifting.floor import detect_floor
from pivot.lifting.objects import lift_track
from pivot.perception.tracks import Track
from pivot.reconstruction.types import Reconstruction
from pivot.scene.graph import STRUCTURAL_LABELS, Scene, SceneObject


@dataclass(frozen=True)
class BuildConfig:
    """Parameters of scene assembly.

    Attributes:
        voxel: Object fusion voxel size, in meters.
        min_points: Objects with fewer fused voxels are discarded.
        floor_snap: Objects whose bottom is this close to the floor are set
            exactly on it, in meters.
        floor_band: Points below this height count as floor, in meters.
        floor_resolution: Floor map cell size, in meters.
    """

    voxel: float = 0.03
    min_points: int = 20
    floor_snap: float = 0.10
    floor_band: float = 0.05
    floor_resolution: float = 0.05


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


def _snap_to_floor(box: OrientedBox, snap: float) -> OrientedBox:
    """Extends a box down to the floor when it nearly touches it.

    Legs and bottoms are often occluded, so fitted boxes float slightly.

    Args:
        box: Fitted box.
        snap: Maximum gap to close, in meters.

    Returns:
        The box, extended to z = 0 if its bottom was within ``snap``.
    """
    if not 0.0 < box.z_min <= snap:
        return box
    size = box.size.copy()
    size[2] = box.z_max
    center = box.center.copy()
    center[2] = 0.5 * box.z_max
    return OrientedBox(center, size, box.yaw)


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

    objects: list[SceneObject] = []
    object_points: dict[int, FloatArray] = {}
    for track in tracks:
        cloud = lift_track(leveled, track, voxel=config.voxel)
        if len(cloud) < config.min_points:
            continue
        box = _snap_to_floor(fit_box(cloud), config.floor_snap)
        movable = track.label.lower() not in STRUCTURAL_LABELS
        objects.append(SceneObject(track.track_id, track.label, box, movable, track.score))
        object_points[track.track_id] = cloud

    floor = _floor_map(points, objects, config)
    metadata = {"frames": len(recon), "tracks": len(tracks)}
    return BuiltScene(Scene.from_objects(objects, floor, metadata), leveled, object_points)


def _floor_map(points: FloatArray, objects: list[SceneObject], config: BuildConfig) -> FloorMap:
    """Builds the walkable floor map of a leveled scene.

    Floor hidden under furniture is still floor: the footprints of movable
    objects standing on it are added so that they can be moved elsewhere and
    other objects can take their place.

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
    return FloorMap.from_points(np.vstack([floor_xy, *hidden]), config.floor_resolution)


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
