"""Doors and windows: set in the walls, one per opening.

Depth seen through a window or an open door lands outside the room, so the
points of an opening spread on both sides of its wall and its fitted box ends
up diagonal and meters deep. An opening lies in a wall: it is snapped to the
nearest wall of the room, keeps only its points near that wall, and becomes a
thin box along it. Openings found twice (a glazed door also segmented as a
window) are merged, and slivers narrower than any real opening are dropped.

Walls are taken as the sides of the floor's rectangle, in the room's
orientation, which is right for one rectangular room. An opening far from
those sides (a door between two rooms of a home) goes instead into the
nearest wall found on the floor map: a strip without floor, with floor on at
least one side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.geometry.floor import FloorMap
from pivot.geometry.types import FloatArray
from pivot.scene.graph import SceneObject, is_opening


# An opening farther than this from the floor rectangle's sides is set in an
# inner wall, when one is found on the floor map.
OUTER_WALL_REACH = 0.6
# How far from an opening an inner wall is looked for, and how long a stretch
# of it is examined, in meters.
_INNER_SEARCH = 0.8
_INNER_SPAN = 1.0
# Share of a wall stretch without floor (a doorway leaves a gap), and largest
# share of floor-less cells on its floor side.
_WALL_SOLID = 0.4
_ROOM_OPEN = 0.3
# The floor side of a wall is looked at this many cells away from it, times three.
_WALL_STEP_CELLS = 2


@dataclass(frozen=True)
class Walls:
    """The walls of a room: the sides of its floor rectangle, and inner walls.

    Attributes:
        yaw: Room orientation, in radians.
        low: (2,) minimum floor coordinates in the room frame, in meters.
        high: (2,) maximum floor coordinates in the room frame, in meters.
        floor: Floor map to find inner walls on; outer walls only if None.
    """

    yaw: float
    low: FloatArray
    high: FloatArray
    floor: FloorMap | None = None

    @classmethod
    def from_floor(cls, floor_xy: FloatArray, yaw: float, floor: FloorMap | None = None,
                   quantile: float = 0.005) -> Walls:
        """Finds the walls from floor points.

        Args:
            floor_xy: (N, 2) floor points, in meters.
            yaw: Room orientation, in radians.
            floor: Floor map with the walls taken off (see
                :func:`~pivot.lifting.floor.remove_standing`), to find inner
                walls on; built from the points if None.
            quantile: Fraction of points ignored on each side (floor seen
                through doors and windows).

        Returns:
            The walls.
        """
        local = floor_xy @ rotation_2d(yaw)  # World to room frame.
        return cls(yaw, np.quantile(local, quantile, axis=0),
                   np.quantile(local, 1.0 - quantile, axis=0),
                   floor if floor is not None else FloorMap.from_points(floor_xy, resolution=0.05))

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
        distance, axis, coordinate = min(sides)
        if distance > OUTER_WALL_REACH:
            inner = self._inner_wall(local)
            if inner is not None:
                return inner
        return axis, coordinate

    def _floor_share(self, local: FloatArray) -> FloatArray:
        """Tells which room-frame points lie on the floor map.

        Args:
            local: (..., 2) room-frame points.

        Returns:
            A boolean array, True on the floor.
        """
        assert self.floor is not None
        world = local @ rotation_2d(self.yaw).T
        cells = np.floor((world - self.floor.origin) / self.floor.resolution).astype(int)
        rows, cols = cells[..., 1], cells[..., 0]
        inside = ((rows >= 0) & (rows < self.floor.shape[0])
                  & (cols >= 0) & (cols < self.floor.shape[1]))
        on_floor = np.zeros(rows.shape, dtype=bool)
        on_floor[inside] = self.floor.floor[rows[inside], cols[inside]]
        return on_floor

    def _inner_wall(self, local: FloatArray) -> tuple[int, float] | None:
        """Finds the inner wall nearest to a room-frame point, on the floor map.

        Args:
            local: (2,) room-frame point.

        Returns:
            The room-frame axis the wall is perpendicular to, and its
            coordinate; None if no wall is near.
        """
        if self.floor is None:
            return None
        step = self.floor.resolution
        offsets = np.arange(-_INNER_SEARCH, _INNER_SEARCH + step, step)
        along = np.arange(-_INNER_SPAN, _INNER_SPAN + step, step)
        found: list[tuple[float, int, float]] = []
        side = 3 * _WALL_STEP_CELLS
        for axis in (0, 1):
            # Points of strips across the axis, one strip per offset.
            grid = np.zeros((len(offsets), len(along), 2))
            grid[..., axis] = local[axis] + offsets[:, None]
            grid[..., 1 - axis] = local[1 - axis] + along[None, :]
            solid = 1.0 - self._floor_share(grid).mean(axis=1)
            for k in np.argsort(np.abs(offsets)):
                if solid[k] < _WALL_SOLID or solid[k] >= 0.999:
                    continue
                if min(solid[max(k - side, 0)], solid[min(k + side, len(offsets) - 1)]) \
                        <= _ROOM_OPEN:
                    found.append((abs(float(offsets[k])), axis, float(local[axis] + offsets[k])))
                    break
        if not found:
            return None
        _, axis, coordinate = min(found)
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
