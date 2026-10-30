"""Gravity-aligned oriented bounding boxes.

All boxes live in a z-up metric world frame where the floor is the plane z = 0.
A box may only rotate about the vertical axis (yaw), which matches how furniture
rests on a floor and keeps collision tests exact and cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from pivot.geometry.types import BoolArray, FloatArray


def rotation_2d(yaw: float) -> FloatArray:
    """Returns the 2x2 rotation matrix for a counter-clockwise angle.

    Args:
        yaw: Rotation angle about +z, in radians.

    Returns:
        A (2, 2) rotation matrix.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def wrap_angle(angle: float) -> float:
    """Wraps an angle to the interval [-pi, pi).

    Args:
        angle: Angle in radians.

    Returns:
        The equivalent angle in [-pi, pi).
    """
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class OrientedBox:
    """A box rotated about the vertical axis only.

    Attributes:
        center: (3,) box center in world coordinates, in meters.
        size: (3,) full extents along the box's local x, y and z axes, in meters.
        yaw: Rotation of the local x axis about +z, in radians.
    """

    center: FloatArray
    size: FloatArray
    yaw: float = 0.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64).reshape(3)
        size = np.asarray(self.size, dtype=np.float64).reshape(3)
        if np.any(size < 0):
            raise ValueError(f"Box size must be non-negative, got {size}.")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "size", size)
        object.__setattr__(self, "yaw", wrap_angle(float(self.yaw)))

    @property
    def z_min(self) -> float:
        """Height of the bottom face."""
        return float(self.center[2] - 0.5 * self.size[2])

    @property
    def z_max(self) -> float:
        """Height of the top face."""
        return float(self.center[2] + 0.5 * self.size[2])

    @property
    def volume(self) -> float:
        """Box volume in cubic meters."""
        return float(np.prod(self.size))

    @property
    def footprint_area(self) -> float:
        """Area of the box's projection onto the floor, in square meters."""
        return float(self.size[0] * self.size[1])

    def footprint(self) -> FloatArray:
        """Returns the floor projection of the box.

        Returns:
            A (4, 2) array of corners in counter-clockwise order.
        """
        hx, hy = 0.5 * self.size[0], 0.5 * self.size[1]
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
        result: FloatArray = local @ rotation_2d(self.yaw).T + self.center[:2]
        return result

    def corners(self) -> FloatArray:
        """Returns the eight corners of the box.

        Returns:
            An (8, 3) array: the four bottom corners, then the four top corners,
            both in counter-clockwise order.
        """
        footprint = self.footprint()
        bottom = np.column_stack([footprint, np.full(4, self.z_min)])
        top = np.column_stack([footprint, np.full(4, self.z_max)])
        return np.vstack([bottom, top])

    def to_local(self, points: FloatArray) -> FloatArray:
        """Expresses world points in the box frame.

        Args:
            points: (N, 3) world points.

        Returns:
            (N, 3) points relative to the box center, in box axes.
        """
        offset = np.asarray(points, dtype=np.float64) - self.center
        xy = offset[:, :2] @ rotation_2d(self.yaw)
        return np.column_stack([xy, offset[:, 2]])

    def contains(self, points: FloatArray, margin: float = 0.0) -> BoolArray:
        """Tests which points lie inside the box.

        Args:
            points: (N, 3) world points.
            margin: Distance by which the box is inflated on every side.

        Returns:
            (N,) boolean mask of points inside the inflated box.
        """
        local = np.abs(self.to_local(points))
        inside: BoolArray = np.all(local <= 0.5 * self.size + margin, axis=1)
        return inside

    def moved(
        self, center_xy: FloatArray | None = None, yaw: float | None = None,
        z_min: float | None = None,
    ) -> OrientedBox:
        """Returns a copy of the box at a new pose, keeping its size.

        Args:
            center_xy: New (2,) floor position of the center. Unchanged if None.
            yaw: New yaw in radians. Unchanged if None.
            z_min: New height of the bottom face. Unchanged if None.

        Returns:
            The moved box.
        """
        center = self.center.copy()
        if center_xy is not None:
            center[:2] = np.asarray(center_xy, dtype=np.float64).reshape(2)
        if z_min is not None:
            center[2] = z_min + 0.5 * self.size[2]
        return OrientedBox(center, self.size, self.yaw if yaw is None else yaw)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the box to JSON-compatible types.

        Returns:
            A dictionary with ``center``, ``size`` and ``yaw`` entries.
        """
        return {"center": self.center.tolist(), "size": self.size.tolist(), "yaw": self.yaw}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrientedBox:
        """Builds a box from the output of :meth:`to_dict`.

        Args:
            data: Dictionary with ``center``, ``size`` and ``yaw`` entries.

        Returns:
            The deserialized box.
        """
        return cls(np.asarray(data["center"]), np.asarray(data["size"]), float(data["yaw"]))


def min_area_rectangle(points_xy: FloatArray) -> tuple[FloatArray, FloatArray, float]:
    """Finds the minimum-area rectangle enclosing a 2D point set.

    The optimal rectangle has one side collinear with an edge of the convex hull,
    so only hull edge directions need to be tested.

    Args:
        points_xy: (N, 2) points, N >= 1.

    Returns:
        A tuple ``(center, size, yaw)`` with the (2,) rectangle center, its (2,)
        side lengths, and the orientation of its first side in radians.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    try:
        hull = points_xy[ConvexHull(points_xy).vertices]
    except (QhullError, ValueError):
        # Degenerate input (fewer than 3 points or collinear): fall back to an
        # axis-aligned rectangle, which is optimal in that case up to rotation.
        lo, hi = points_xy.min(axis=0), points_xy.max(axis=0)
        return 0.5 * (lo + hi), hi - lo, 0.0

    edges = np.roll(hull, -1, axis=0) - hull
    # Rectangle orientations are defined modulo pi/2.
    angles = np.unique(np.mod(np.arctan2(edges[:, 1], edges[:, 0]), 0.5 * math.pi))
    # Hull coordinates in every candidate frame: (angles, vertices, 2).
    local = np.einsum("vj,ajk->avk", hull, np.stack([rotation_2d(float(a)) for a in angles]))
    lo, hi = local.min(axis=1), local.max(axis=1)
    best = int(np.argmin(np.prod(hi - lo, axis=1)))
    angle = float(angles[best])
    center = rotation_2d(angle) @ (0.5 * (lo[best] + hi[best]))
    return center, hi[best] - lo[best], angle


def fit_box(points: FloatArray, yaw: float | None = None, trim: float = 0.0) -> OrientedBox:
    """Fits a gravity-aligned box around a point cloud.

    Args:
        points: (N, 3) world points, N >= 1.
        yaw: Orientation to fit the box at, in radians; the minimum-area
            orientation if None.
        trim: Fraction of the points ignored at each end of every axis, so
            that a few stray points do not inflate the box. 0 encloses every
            point.

    Returns:
        The fitted box. Its longer horizontal side is along the local x axis.

    Raises:
        ValueError: If ``points`` is empty.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("Cannot fit a box to an empty point set.")
    if yaw is None:
        _, _, yaw = min_area_rectangle(points[:, :2])
        if trim > 0.0:
            # Stray points can rotate the hull: re-estimate on the kept ones.
            local = points[:, :2] @ rotation_2d(yaw)
            lo, hi = np.quantile(local, [trim, 1.0 - trim], axis=0)
            kept = np.all((local >= lo) & (local <= hi), axis=1)
            if kept.sum() >= 3:
                _, _, yaw = min_area_rectangle(points[kept, :2])
    # Box-local coordinates: x along yaw, y across it.
    local = np.column_stack([points[:, :2] @ rotation_2d(yaw), points[:, 2]])
    lo, hi = np.quantile(local, [trim, 1.0 - trim], axis=0)
    size = hi - lo
    center = np.concatenate([rotation_2d(yaw) @ (0.5 * (lo[:2] + hi[:2])), [0.5 * (lo[2] + hi[2])]])
    if size[1] > size[0]:
        size[:2] = size[1::-1]
        yaw += 0.5 * math.pi
    return OrientedBox(center, size, wrap_angle(yaw))
