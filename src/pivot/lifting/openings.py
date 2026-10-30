"""Doors and windows: set in the walls, one per opening.

Depth seen through a window or an open door lands outside the room, so the
points of an opening spread on both sides of its wall and its fitted box ends
up diagonal and meters deep. An opening lies in a wall: it is snapped to the
nearest wall of the room, keeps only its points near that wall, and becomes a
thin box along it. Openings found twice (a glazed door also segmented as a
window) are merged, and slivers narrower than any real opening are dropped.

Walls are taken as the sides of the room's floor rectangle, in the room's
orientation: right for rectangular rooms, approximate for others.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.geometry.types import FloatArray
from pivot.scene.graph import SceneObject, is_opening


@dataclass(frozen=True)
class Walls:
    """The four walls of a room, as the sides of its floor rectangle.

    Attributes:
        yaw: Room orientation, in radians.
        low: (2,) minimum floor coordinates in the room frame, in meters.
        high: (2,) maximum floor coordinates in the room frame, in meters.
    """

    yaw: float
    low: FloatArray
    high: FloatArray

    @classmethod
    def from_floor(cls, floor_xy: FloatArray, yaw: float, quantile: float = 0.005) -> Walls:
        """Finds the walls from floor points.

        Args:
            floor_xy: (N, 2) floor points, in meters.
            yaw: Room orientation, in radians.
            quantile: Fraction of points ignored on each side (floor seen
                through doors and windows).

        Returns:
            The walls.
        """
        local = floor_xy @ rotation_2d(yaw)  # World to room frame.
        return cls(yaw, np.quantile(local, quantile, axis=0),
                   np.quantile(local, 1.0 - quantile, axis=0))

    def to_room(self, xy: FloatArray) -> FloatArray:
        """Expresses world points in the room frame.

        Args:
            xy: (N, 2) world points.

        Returns:
            (N, 2) room-frame points.
        """
        return xy @ rotation_2d(self.yaw)

    def nearest(self, xy: FloatArray) -> tuple[int, float]:
        """Finds the wall nearest to a point.

        Args:
            xy: (2,) world point.

        Returns:
            The room-frame axis the wall is perpendicular to (0 or 1), and the
            wall's coordinate along that axis.
        """
        local = self.to_room(xy[None])[0]
        sides = [(abs(local[a] - c), a, float(c))
                 for a in (0, 1) for c in (self.low[a], self.high[a])]
        _, axis, coordinate = min(sides)
        return axis, coordinate


def snap_opening(obj: SceneObject, points: FloatArray, walls: Walls, band: float = 0.25,
                 thickness: float = 0.1, min_points: int = 30) -> SceneObject | None:
    """Sets an opening in its nearest wall.

    Args:
        obj: The door or window.
        points: (N, 3) its fused points.
        walls: The room's walls.
        band: Points farther than this from the wall are dropped, in meters.
        thickness: Thickness of the snapped box, in meters.
        min_points: Openings with fewer points near the wall are dropped.

    Returns:
        The opening with a thin box along the wall, or None if too few of its
        points lie near a wall.
    """
    axis, coordinate = walls.nearest(np.median(points[:, :2], axis=0))
    local = walls.to_room(points[:, :2])
    near = np.abs(local[:, axis] - coordinate) <= band
    if near.sum() < min_points:
        return None
    span = np.quantile(local[near, 1 - axis], [0.02, 0.98])
    heights = np.quantile(points[near, 2], [0.02, 0.98])
    return replace(obj, box=_wall_box(walls, axis, coordinate, span, heights, thickness),
                   shape=None)


def _wall_box(walls: Walls, axis: int, coordinate: float, span: FloatArray,
              heights: FloatArray, thickness: float) -> OrientedBox:
    """Builds a thin box lying in a wall.

    Args:
        walls: The room's walls.
        axis: Room-frame axis the wall is perpendicular to.
        coordinate: The wall's coordinate along that axis, in meters.
        span: (2,) extent along the wall, in room-frame meters.
        heights: (2,) bottom and top, in meters.
        thickness: Box thickness across the wall, in meters.

    Returns:
        The box; its first axis runs along the wall.
    """
    center_local = np.zeros(2)
    center_local[axis] = coordinate
    center_local[1 - axis] = 0.5 * (span[0] + span[1])
    center_xy = rotation_2d(walls.yaw) @ center_local
    yaw = walls.yaw + (0.5 * math.pi if axis == 0 else 0.0)
    return OrientedBox(np.array([*center_xy, 0.5 * (heights[0] + heights[1])]),
                       np.array([span[1] - span[0], thickness, heights[1] - heights[0]]), yaw)


def _same_opening(a: OrientedBox, b: OrientedBox, min_overlap: float) -> bool:
    """Tests whether two snapped openings are the same one.

    Args:
        a: First snapped box.
        b: Second snapped box.
        min_overlap: Fraction of the narrower opening that must overlap the
            other, along the wall and in height.

    Returns:
        True if they share a wall and mostly overlap.
    """
    if abs(math.remainder(a.yaw - b.yaw, math.pi)) > 1e-3:
        return False
    offset = rotation_2d(-a.yaw) @ (b.center[:2] - a.center[:2])
    if abs(offset[1]) > a.size[1]:
        return False
    overlap_along = min(0.5 * a.size[0], offset[0] + 0.5 * b.size[0]) - max(
        -0.5 * a.size[0], offset[0] - 0.5 * b.size[0])
    overlap_z = min(a.z_max, b.z_max) - max(a.z_min, b.z_min)
    return bool(overlap_along >= min_overlap * min(a.size[0], b.size[0])
                and overlap_z >= min_overlap * min(a.size[2], b.size[2]))


def clean_openings(objects: list[SceneObject], points: dict[int, FloatArray], walls: Walls,
                   min_width: float = 0.3, min_overlap: float = 0.5) -> list[SceneObject]:
    """Snaps every opening to its wall, merges duplicates and drops slivers.

    Of two openings found at the same place, the one with more points is
    kept, with the union of both extents.

    Args:
        objects: All objects; those that are not openings are kept as they are.
        points: Fused points of every object, keyed by id.
        walls: The room's walls.
        min_width: Openings narrower than this along their wall are dropped, in meters.
        min_overlap: See :func:`_same_opening`.

    Returns:
        The objects, with clean openings.
    """
    others = [o for o in objects if not is_opening(o)]
    snapped = [s for o in objects if is_opening(o)
               if (s := snap_opening(o, points[o.object_id], walls)) is not None
               and s.box.size[0] >= min_width]
    snapped.sort(key=lambda o: -len(points[o.object_id]))
    kept: list[SceneObject] = []
    for opening in snapped:
        match = next((k for k in kept if _same_opening(k.box, opening.box, min_overlap)), None)
        if match is None:
            kept.append(opening)
        else:
            kept[kept.index(match)] = replace(match, box=_union_along(match.box, opening.box))
    return others + sorted(kept, key=lambda o: o.object_id)


def _union_along(a: OrientedBox, b: OrientedBox) -> OrientedBox:
    """Unites two openings on one wall, keeping the first one's wall position.

    Args:
        a: The kept opening.
        b: The merged one.

    Returns:
        The box spanning both, along the wall and in height.
    """
    offset = rotation_2d(-a.yaw) @ (b.center[:2] - a.center[:2])
    lo = min(-0.5 * a.size[0], offset[0] - 0.5 * b.size[0])
    hi = max(0.5 * a.size[0], offset[0] + 0.5 * b.size[0])
    z_lo, z_hi = min(a.z_min, b.z_min), max(a.z_max, b.z_max)
    center_xy = a.center[:2] + rotation_2d(a.yaw) @ np.array([0.5 * (lo + hi), 0.0])
    return OrientedBox(np.array([*center_xy, 0.5 * (z_lo + z_hi)]),
                       np.array([hi - lo, a.size[1], z_hi - z_lo]), a.yaw)
