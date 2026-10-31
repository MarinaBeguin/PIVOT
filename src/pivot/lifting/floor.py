"""Floor detection (the reference plane every other measurement relies on), walls left on
the floor map, and ceiling height."""

import dataclasses

import numpy as np

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.geometry.floor import FloorMap
from pivot.geometry.plane import Plane, fit_plane_ransac, floor_alignment
from pivot.geometry.types import FloatArray, IntArray


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


def estimate_ceiling(points: FloatArray, floor: FloorMap | None = None, min_height: float = 2.0,
                     bin_size: float = 0.05, min_peak_ratio: float = 3.0,
                     edge_fraction: float = 0.25) -> float | None:
    """Estimates the ceiling height of a leveled scene, if the video saw the ceiling.

    Points are first reduced to one per ``bin_size`` cube, so that counts
    measure surface area rather than how long the camera looked. A ceiling is
    then a horizontal layer: a peak of the height histogram well above the
    background of the walls. Without such a peak (the camera never looked up)
    the ceiling is unknown, rather than mistaken for the top of the walls.

    Args:
        points: (N, 3) leveled scene points, floor at z = 0.
        floor: The room's floor map: only points above it count, so that what
            is seen through the windows (buildings, trees) does not drown the
            ceiling; all points if None.
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
    if floor is not None:
        cells = np.floor((high[:, :2] - floor.origin) / floor.resolution).astype(np.int64)
        rows, cols = floor.shape
        inside = ((cells[:, 0] >= 0) & (cells[:, 0] < cols)
                  & (cells[:, 1] >= 0) & (cells[:, 1] < rows))
        above = np.zeros(len(high), dtype=bool)
        above[inside] = floor.floor[cells[inside, 1], cells[inside, 0]]
        high = high[above]
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


# A cell where floor was seen holds at least this share of a typical floor cell's points.
SEEN_FLOOR_SHARE = 0.25
# Walls are looked for between these heights (above skirting boards and floor
# clutter, below the ceiling), in levels of this height, and must have points
# in at least this many levels.
WALL_LOW, WALL_HIGH, WALL_LEVEL, WALL_MIN_LEVELS = 0.3, 1.8, 0.1, 8
# Objects' boxes are grown by this much before their points are set aside.
_BOX_MARGIN = 0.05


def remove_standing(floor: FloorMap, points: FloatArray, boxes: list[OrientedBox],
                    floor_band: float = 0.05) -> FloorMap:
    """Undoes the gap filling of the floor map where something stands: thin walls.

    The floor map fills small gaps in the observed floor, which also erases
    thin walls: no floor is seen under a 10 cm partition between two rooms,
    and the filling closes it. Such filled cells are taken off again where
    points stand above them at many heights (:data:`WALL_MIN_LEVELS`),
    outside every object's box. Cells where floor was actually seen stay floor.

    Args:
        floor: The floor map.
        points: (N, 3) leveled scene points, floor at z = 0 (the floor's
            points say where it was seen).
        boxes: Boxes of the segmented objects; their points are not walls.
        floor_band: Points below this height are floor, in meters.

    Returns:
        The floor map without those cells.
    """
    # Floor seen from many frames is dense; the foot of a wall leaves only a few points.
    floor_counts = _cell_counts(floor, points[np.abs(points[:, 2]) < floor_band])
    seen = np.zeros(floor.shape, dtype=bool)
    if (floor_counts > 0).any():
        seen = floor_counts >= SEEN_FLOOR_SHARE * float(np.median(floor_counts[floor_counts > 0]))
    standing = _outside_boxes(points[(points[:, 2] >= WALL_LOW) & (points[:, 2] <= WALL_HIGH)],
                              boxes)
    # Distinct height levels with points, per cell.
    cells = np.floor((standing[:, :2] - floor.origin) / floor.resolution)
    levels = np.unique(np.column_stack([cells, np.floor(standing[:, 2] / WALL_LEVEL)]), axis=0)
    counts = _cell_counts(floor, floor.origin + (levels[:, :2] + 0.5) * floor.resolution)
    return dataclasses.replace(floor, floor=floor.floor & (seen | (counts < WALL_MIN_LEVELS)))


def _outside_boxes(points: FloatArray, boxes: list[OrientedBox]) -> FloatArray:
    """Keeps the points outside every box (grown by :data:`_BOX_MARGIN`).

    Args:
        points: (N, 3) points.
        boxes: The boxes.

    Returns:
        The points no box holds.
    """
    keep = np.ones(len(points), dtype=bool)
    for box in boxes:
        local = (points[:, :2] - box.center[:2]) @ rotation_2d(box.yaw)
        inside = np.all(np.abs(local) <= 0.5 * box.size[:2] + _BOX_MARGIN, axis=1)
        heights = points[:, 2]
        inside &= (heights >= box.z_min - _BOX_MARGIN) & (heights <= box.z_max + _BOX_MARGIN)
        keep &= ~inside
    kept: FloatArray = points[keep]
    return kept


def _cell_counts(floor: FloorMap, xy: FloatArray) -> IntArray:
    """Counts points per floor map cell.

    Args:
        floor: The floor map.
        xy: (N, 2+) points; only their first two coordinates are used.

    Returns:
        (rows, cols) counts; points off the grid are ignored.
    """
    cells = np.floor((xy[:, :2] - floor.origin) / floor.resolution).astype(np.int64)
    rows, cols = floor.shape
    inside = (cells[:, 0] >= 0) & (cells[:, 0] < cols) & (cells[:, 1] >= 0) & (cells[:, 1] < rows)
    counts = np.zeros(floor.shape, dtype=np.int64)
    np.add.at(counts, (cells[inside, 1], cells[inside, 0]), 1)
    return counts
