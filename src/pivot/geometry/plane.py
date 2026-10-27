"""Plane fitting and gravity alignment of reconstructed scenes.

Video reconstructions come out in an arbitrary camera-defined frame. Everything
downstream assumes a z-up world whose floor is z = 0, so the floor plane is
detected once and the whole scene is rigidly transformed onto it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pivot.geometry.types import BoolArray, FloatArray


@dataclass(frozen=True)
class Plane:
    """A plane ``normal . p + offset = 0`` with a unit normal.

    Attributes:
        normal: (3,) unit normal.
        offset: Signed offset along the normal.
    """

    normal: FloatArray
    offset: float

    def signed_distance(self, points: FloatArray) -> FloatArray:
        """Computes the signed distances of points to the plane.

        Args:
            points: (N, 3) points.

        Returns:
            (N,) distances, positive on the side the normal points to.
        """
        result: FloatArray = np.asarray(points, dtype=np.float64) @ self.normal + self.offset
        return result

    def flipped(self) -> Plane:
        """Returns the same plane with its normal reversed.

        Returns:
            The flipped plane.
        """
        return Plane(-self.normal, -self.offset)


def fit_plane_lstsq(points: FloatArray) -> Plane:
    """Fits a plane to points in the total least squares sense.

    Args:
        points: (N, 3) points, N >= 3.

    Returns:
        The best-fit plane.
    """
    centroid = points.mean(axis=0)
    # The normal is the direction of least variance.
    _, _, vt = np.linalg.svd(points - centroid, full_matrices=False)
    normal = vt[-1]
    return Plane(normal, float(-normal @ centroid))


def fit_plane_ransac(
    points: FloatArray,
    threshold: float,
    iterations: int = 500,
    rng: np.random.Generator | None = None,
    allowed: BoolArray | None = None,
) -> tuple[Plane, BoolArray]:
    """Robustly fits a plane with RANSAC, then refines it on its inliers.

    Args:
        points: (N, 3) points.
        threshold: Maximum point-to-plane distance of an inlier.
        iterations: Number of random 3-point hypotheses.
        rng: Random generator, for reproducibility.
        allowed: Optional (N,) mask of points that may be sampled as hypotheses.
            Inliers are still counted over all points.

    Returns:
        A tuple ``(plane, inliers)`` with the refined plane and its (N,) inlier mask.

    Raises:
        ValueError: If fewer than three candidate points are available.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    candidates = np.flatnonzero(allowed) if allowed is not None else np.arange(len(points))
    if len(candidates) < 3:
        raise ValueError("RANSAC needs at least three candidate points.")

    best_inliers = np.zeros(len(points), dtype=bool)
    for _ in range(iterations):
        sample = points[rng.choice(candidates, size=3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        normal /= norm
        inliers = np.abs(points @ normal - normal @ sample[0]) < threshold
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    plane = fit_plane_lstsq(points[best_inliers])
    inliers = np.abs(plane.signed_distance(points)) < threshold
    return plane, inliers


def rotation_between(source: FloatArray, target: FloatArray) -> FloatArray:
    """Returns the smallest rotation mapping one unit vector onto another.

    Args:
        source: (3,) unit vector.
        target: (3,) unit vector.

    Returns:
        A (3, 3) rotation matrix ``R`` with ``R @ source == target``.
    """
    axis = np.cross(source, target)
    sin = float(np.linalg.norm(axis))
    cos = float(source @ target)
    if sin < 1e-12:
        if cos > 0:
            return np.eye(3)
        # Opposite vectors: rotate by pi about any axis orthogonal to source.
        ortho = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, ortho)
        axis /= np.linalg.norm(axis)
        result: FloatArray = 2.0 * np.outer(axis, axis) - np.eye(3)
        return result
    k = axis / sin
    skew = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    rotation: FloatArray = np.eye(3) + sin * skew + (1.0 - cos) * skew @ skew
    return rotation


def floor_alignment(floor: Plane, up_hint: FloatArray) -> FloatArray:
    """Computes the rigid transform that puts a floor plane at z = 0, z up.

    Args:
        floor: The detected floor plane, in any orientation.
        up_hint: (3,) approximate up direction in the source frame, used to
            resolve the sign of the normal (e.g. the mean camera up vector).

    Returns:
        A (4, 4) homogeneous transform from the source frame to the aligned frame.
    """
    plane = floor if floor.normal @ up_hint >= 0 else floor.flipped()
    rotation = rotation_between(plane.normal, np.array([0.0, 0.0, 1.0]))
    transform = np.eye(4)
    transform[:3, :3] = rotation
    # A point on the plane, -offset * normal, must land at z = 0.
    transform[2, 3] = plane.offset
    return transform


def apply_transform(transform: FloatArray, points: FloatArray) -> FloatArray:
    """Applies a homogeneous transform to points.

    Args:
        transform: (4, 4) homogeneous transform.
        points: (N, 3) points.

    Returns:
        (N, 3) transformed points.
    """
    result: FloatArray = points @ transform[:3, :3].T + transform[:3, 3]
    return result
