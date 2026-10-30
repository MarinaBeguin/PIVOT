"""Floor detection (the reference plane every other measurement relies on) and ceiling height."""

import numpy as np

from pivot.geometry.plane import Plane, fit_plane_ransac, floor_alignment
from pivot.geometry.types import FloatArray


def detect_floor(points: FloatArray, up_hint: FloatArray, threshold: float = 0.03,
                 low_band: float = 20.0, max_tilt_deg: float = 15.0,
                 rng: np.random.Generator | None = None) -> tuple[Plane, FloatArray]:
    """Finds the floor plane and the transform that levels the scene on it.

    The floor is the dominant plane among the lowest points along the up hint.
    Restricting hypotheses to that band avoids locking onto a wall or a table.

    Args:
        points: (N, 3) scene points in the reconstruction frame.
        up_hint: (3,) approximate up direction in that frame.
        threshold: RANSAC inlier distance, in meters.
        low_band: Percentage of lowest points allowed to seed hypotheses.
        max_tilt_deg: Maximum angle between the floor normal and the up hint.
        rng: Random generator, for reproducibility.

    Returns:
        A tuple ``(plane, transform)``: the floor plane in the input frame and
        the (4, 4) transform to the leveled frame where the floor is z = 0.

    Raises:
        ValueError: If no sufficiently level plane is found.
    """
    heights = points @ up_hint
    allowed = heights <= np.percentile(heights, low_band)
    plane, _ = fit_plane_ransac(points, threshold, rng=rng, allowed=allowed)
    tilt = np.degrees(np.arccos(min(1.0, abs(float(plane.normal @ up_hint)))))
    if tilt > max_tilt_deg:
        raise ValueError(f"Best low plane is tilted by {tilt:.0f} degrees; no floor found.")
    return plane, floor_alignment(plane, up_hint)


def room_orientation(points: FloatArray, min_height: float = 0.5, max_height: float = 2.5,
                     bin_size: float = 0.02, step_deg: float = 0.5) -> float:
    """Estimates the orientation of a room's walls in a leveled scene.

    Walls are vertical planes, so the horizontal coordinates of points at wall
    height pile up into sharp peaks when projected on the axis normal to a
    wall. The orientation is the one, modulo 90 degrees, that makes the
    projection histograms the most peaked (largest sum of squared counts).

    Args:
        points: (N, 3) leveled scene points.
        min_height: Lowest height considered, above most furniture, in meters.
        max_height: Highest height considered, below the ceiling, in meters.
        bin_size: Histogram bin size, in meters.
        step_deg: Angular resolution of the search, in degrees.

    Returns:
        The wall orientation in [0, pi/2), in radians.
    """
    xy = points[(points[:, 2] > min_height) & (points[:, 2] < max_height), :2]
    xy = xy - xy.mean(axis=0)
    angles = np.radians(np.arange(0.0, 90.0, step_deg))
    scores = []
    for angle in angles:
        direction = np.array([np.cos(angle), np.sin(angle)])
        score = 0.0
        # Both wall families of a rectangular room: along and across the direction.
        for axis in (direction, np.array([-direction[1], direction[0]])):
            counts = np.bincount(np.floor((xy @ axis) / bin_size - (xy @ axis).min() / bin_size)
                                 .astype(np.int64))
            score += float(np.sum(counts.astype(np.float64) ** 2))
        scores.append(score)
    return float(angles[int(np.argmax(scores))])


def estimate_ceiling(points: FloatArray, min_height: float = 2.0, bin_size: float = 0.05,
                     min_peak_ratio: float = 3.0, edge_fraction: float = 0.25) -> float | None:
    """Estimates the ceiling height of a leveled scene, if the video saw the ceiling.

    Points are first reduced to one per ``bin_size`` cube, so that counts
    measure surface area rather than how long the camera looked. A ceiling is
    then a horizontal layer: a peak of the height histogram well above the
    background of the walls. Without such a peak (the camera never looked up)
    the ceiling is unknown, rather than mistaken for the top of the walls.

    Args:
        points: (N, 3) leveled scene points, floor at z = 0.
        min_height: Lowest height a ceiling can have, in meters.
        bin_size: Cube and histogram bin size, in meters.
        min_peak_ratio: How many times the median wall bin the peak must hold.
        edge_fraction: The ceiling is the lowest height where the peak's layer
            still holds this fraction of the peak count, so that the estimate
            errs on the low (safe) side.

    Returns:
        The ceiling height in meters, or None if no ceiling layer is found.
    """
    high = points[points[:, 2] >= min_height]
    if len(high) == 0:
        return None
    cells = np.unique(np.floor(high / bin_size).astype(np.int64), axis=0)
    counts = np.bincount(cells[:, 2] - cells[:, 2].min())
    peak = int(np.argmax(counts))
    if peak == 0 or counts[peak] < min_peak_ratio * max(float(np.median(counts[:peak])), 1.0):
        return None
    bottom = peak
    while bottom > 0 and counts[bottom - 1] >= edge_fraction * counts[peak]:
        bottom -= 1
    return round(float((cells[:, 2].min() + bottom) * bin_size), 3)
