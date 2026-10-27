"""Tests for plane fitting and floor alignment."""

import numpy as np
import pytest

from pivot.geometry.plane import (
    Plane,
    apply_transform,
    fit_plane_ransac,
    floor_alignment,
    rotation_between,
)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Returns a uniformly random 3D rotation matrix."""
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_ransac_ignores_outliers() -> None:
    rng = np.random.default_rng(0)
    floor = np.column_stack([rng.uniform(-2, 2, (800, 2)), rng.normal(0, 0.005, 800)])
    clutter = rng.uniform([-2, -2, 0.2], [2, 2, 2.0], (400, 3))
    plane, inliers = fit_plane_ransac(np.vstack([floor, clutter]), threshold=0.02, rng=rng)
    assert abs(plane.normal[2]) == pytest.approx(1.0, abs=1e-3)
    assert inliers[:800].mean() > 0.99
    assert inliers[800:].mean() < 0.02


def test_ransac_requires_three_points() -> None:
    with pytest.raises(ValueError):
        fit_plane_ransac(np.zeros((2, 3)), threshold=0.1)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotation_between_maps_source_to_target(seed: int) -> None:
    rng = np.random.default_rng(seed)
    source, target = (v / np.linalg.norm(v) for v in rng.normal(size=(2, 3)))
    rotation = rotation_between(source, target)
    np.testing.assert_allclose(rotation @ source, target, atol=1e-10)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-10)


def test_rotation_between_opposite_vectors() -> None:
    up = np.array([0.0, 0.0, 1.0])
    rotation = rotation_between(up, -up)
    np.testing.assert_allclose(rotation @ up, -up, atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_floor_alignment_puts_floor_at_zero_with_up_hint() -> None:
    rng = np.random.default_rng(3)
    rotation = random_rotation(rng)
    offset = np.array([0.3, -1.2, 2.0])
    floor = np.column_stack([rng.uniform(-2, 2, (300, 2)), np.zeros(300)])
    above = np.array([[0.0, 0.0, 1.5]])

    def to_source(points: np.ndarray) -> np.ndarray:
        return points @ rotation.T + offset

    plane, _ = fit_plane_ransac(to_source(floor), threshold=0.01, rng=rng)
    # Feed the plane with its normal pointing down to check the sign fix.
    if plane.normal @ (rotation @ [0, 0, 1]) > 0:
        plane = Plane(-plane.normal, -plane.offset)
    transform = floor_alignment(plane, up_hint=rotation @ np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(apply_transform(transform, to_source(floor))[:, 2], 0.0, atol=1e-9)
    assert apply_transform(transform, to_source(above))[0, 2] == pytest.approx(1.5)
