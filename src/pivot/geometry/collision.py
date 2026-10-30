"""Exact distance and overlap tests between convex polygons and boxes.

A gravity-aligned box is a vertical prism: a convex footprint times a height
interval. The distance between two such prisms factorizes as
``sqrt(d_footprint**2 + d_height**2)``, so every 3D query reduces to 2D convex
polygon geometry plus 1D interval arithmetic.
"""

import math

import numpy as np

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.geometry.types import FloatArray

_EPS = 1e-9


def _edge_normals(polygon: FloatArray) -> FloatArray:
    """Returns the unit outward normals of a counter-clockwise convex polygon.

    Args:
        polygon: (N, 2) vertices in counter-clockwise order.

    Returns:
        (N, 2) unit normals, one per edge.
    """
    edges = np.roll(polygon, -1, axis=0) - polygon
    normals = np.column_stack([edges[:, 1], -edges[:, 0]])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    result: FloatArray = normals / np.maximum(lengths, _EPS)
    return result


def polygon_penetration(a: FloatArray, b: FloatArray) -> float:
    """Measures how deeply two convex polygons overlap.

    Uses the separating axis theorem. The result is the smallest translation
    distance along an edge normal that separates the polygons.

    Args:
        a: (N, 2) convex polygon, counter-clockwise.
        b: (M, 2) convex polygon, counter-clockwise.

    Returns:
        The penetration depth in meters, or 0.0 if the polygons do not overlap.
    """
    depth = math.inf
    for normal in np.vstack([_edge_normals(a), _edge_normals(b)]):
        pa, pb = a @ normal, b @ normal
        overlap = min(pa.max(), pb.max()) - max(pa.min(), pb.min())
        if overlap <= 0.0:
            return 0.0
        depth = min(depth, float(overlap))
    return depth


def _point_segment_distance(points: FloatArray, start: FloatArray, end: FloatArray) -> FloatArray:
    """Computes distances from points to a segment.

    Args:
        points: (N, 2) query points.
        start: (2,) segment start.
        end: (2,) segment end.

    Returns:
        (N,) Euclidean distances.
    """
    direction = end - start
    length_sq = float(direction @ direction)
    t = np.zeros(len(points)) if length_sq < _EPS else (points - start) @ direction / length_sq
    closest = start + np.clip(t, 0.0, 1.0)[:, None] * direction
    result: FloatArray = np.linalg.norm(points - closest, axis=1)
    return result


def points_to_polygon_distance(points: FloatArray, polygon: FloatArray) -> FloatArray:
    """Computes distances from points to a convex polygon region.

    Args:
        points: (N, 2) query points.
        polygon: (M, 2) convex polygon, counter-clockwise.

    Returns:
        (N,) distances; 0 for points inside the polygon.
    """
    inside = np.ones(len(points), dtype=bool)
    distances = np.full(len(points), math.inf)
    for i, start in enumerate(polygon):
        end = polygon[(i + 1) % len(polygon)]
        edge, rel = end - start, points - start
        inside &= edge[0] * rel[:, 1] - edge[1] * rel[:, 0] >= 0.0
        distances = np.minimum(distances, _point_segment_distance(points, start, end))
    result: FloatArray = np.where(inside, 0.0, distances)
    return result


def polygon_distance(a: FloatArray, b: FloatArray) -> float:
    """Computes the Euclidean distance between two convex polygons.

    Args:
        a: (N, 2) convex polygon, counter-clockwise.
        b: (M, 2) convex polygon, counter-clockwise.

    Returns:
        The minimum distance between the two regions; 0.0 if they touch or overlap.
    """
    if polygon_penetration(a, b) > 0.0:
        return 0.0
    # For disjoint convex polygons the closest pair always involves a vertex.
    best = math.inf
    for poly, other in ((a, b), (b, a)):
        for i, start in enumerate(poly):
            end = poly[(i + 1) % len(poly)]
            best = min(best, float(_point_segment_distance(other, start, end).min()))
    return best


def interval_gap(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> float:
    """Returns the gap between two closed intervals.

    Args:
        lo_a: Lower bound of the first interval.
        hi_a: Upper bound of the first interval.
        lo_b: Lower bound of the second interval.
        hi_b: Upper bound of the second interval.

    Returns:
        The distance between the intervals, or 0.0 if they intersect.
    """
    return max(0.0, lo_b - hi_a, lo_a - hi_b)


def footprint_overlap(box: OrientedBox, other: OrientedBox, samples: int = 9) -> float:
    """Measures how much of a box's footprint lies over another box's footprint.

    Args:
        box: The box whose footprint is sampled.
        other: The other box.
        samples: Footprint samples per side.

    Returns:
        The fraction in [0, 1].
    """
    u = (np.arange(samples) + 0.5) / samples - 0.5
    grid = np.stack(np.meshgrid(u, u), axis=-1).reshape(-1, 2) * box.size[:2]
    world_xy = grid @ rotation_2d(box.yaw).T + box.center[:2]
    world = np.column_stack([world_xy, np.full(len(grid), other.center[2])])
    local = np.abs(other.to_local(world)[:, :2])
    return float(np.mean(np.all(local <= 0.5 * other.size[:2], axis=1)))


def box_distance_lower_bound(a: OrientedBox, b: OrientedBox) -> float:
    """Bounds the distance between two boxes from below, cheaply.

    Uses the circles circumscribing the footprints and the vertical gap.

    Args:
        a: First box.
        b: Second box.

    Returns:
        A value never larger than :func:`box_distance`.
    """
    d_xy = (math.hypot(*(a.center[:2] - b.center[:2]))
            - 0.5 * math.hypot(*a.size[:2]) - 0.5 * math.hypot(*b.size[:2]))
    return max(d_xy, interval_gap(a.z_min, a.z_max, b.z_min, b.z_max), 0.0)


def box_distance(a: OrientedBox, b: OrientedBox) -> float:
    """Computes the Euclidean distance between two gravity-aligned boxes.

    Args:
        a: First box.
        b: Second box.

    Returns:
        The minimum distance between the boxes; 0.0 if they touch or overlap.
    """
    d_xy = polygon_distance(a.footprint(), b.footprint())
    d_z = interval_gap(a.z_min, a.z_max, b.z_min, b.z_max)
    return math.hypot(d_xy, d_z)


def box_penetration(a: OrientedBox, b: OrientedBox) -> float:
    """Measures how deeply two gravity-aligned boxes overlap.

    Args:
        a: First box.
        b: Second box.

    Returns:
        The smallest translation, horizontal or vertical, that separates the
        boxes, or 0.0 if they do not overlap.
    """
    z_overlap = min(a.z_max, b.z_max) - max(a.z_min, b.z_min)
    if z_overlap <= 0.0:
        return 0.0
    xy = polygon_penetration(a.footprint(), b.footprint())
    return min(xy, z_overlap) if xy > 0.0 else 0.0


def boxes_collide(a: OrientedBox, b: OrientedBox, clearance: float = 0.0,
                  tolerance: float = 1e-3) -> bool:
    """Tests whether two boxes overlap or come closer than a clearance.

    Args:
        a: First box.
        b: Second box.
        clearance: Required free distance between the boxes, in meters.
        tolerance: Overlap depth below which touching boxes are accepted. Boxes
            fitted to noisy point clouds routinely interpenetrate by millimeters.

    Returns:
        True if the boxes are in conflict.
    """
    if clearance > 0.0:
        return box_distance(a, b) < clearance
    return box_penetration(a, b) > tolerance
