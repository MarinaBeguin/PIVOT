"""Floor detection: the reference plane every other measurement relies on."""

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
