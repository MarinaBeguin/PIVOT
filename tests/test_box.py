"""Tests for gravity-aligned oriented boxes."""

import math

import numpy as np
import pytest

from pivot.geometry.box import OrientedBox, fit_box, min_area_rectangle, wrap_angle


def test_footprint_is_counter_clockwise_and_rotated() -> None:
    box = OrientedBox(np.array([1.0, 2.0, 0.5]), np.array([2.0, 1.0, 1.0]), math.pi / 2)
    footprint = box.footprint()
    # A quarter turn swaps the extents.
    np.testing.assert_allclose(footprint.min(axis=0), [0.5, 1.0], atol=1e-12)
    np.testing.assert_allclose(footprint.max(axis=0), [1.5, 3.0], atol=1e-12)
    x, y = footprint[:, 0], footprint[:, 1]
    signed_area = 0.5 * (x @ np.roll(y, -1) - y @ np.roll(x, -1))
    assert signed_area == pytest.approx(2.0)


def test_contains_respects_yaw_and_margin() -> None:
    box = OrientedBox(np.zeros(3), np.array([2.0, 0.2, 1.0]), math.pi / 4)
    on_axis = np.array([[0.6, 0.6, 0.0]])
    off_axis = np.array([[0.6, -0.6, 0.0]])
    assert box.contains(on_axis)[0]
    assert not box.contains(off_axis)[0]
    assert box.contains(off_axis, margin=1.0)[0]


def test_moved_keeps_size_and_sets_height() -> None:
    box = OrientedBox(np.array([0.0, 0.0, 0.4]), np.array([1.0, 1.0, 0.8]))
    moved = box.moved(np.array([3.0, -1.0]), yaw=1.0, z_min=0.75)
    np.testing.assert_allclose(moved.size, box.size)
    assert moved.z_min == pytest.approx(0.75)
    np.testing.assert_allclose(moved.center[:2], [3.0, -1.0])
    assert moved.yaw == pytest.approx(1.0)


def test_round_trip_serialization() -> None:
    box = OrientedBox(np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.4, 0.3]), 0.7)
    restored = OrientedBox.from_dict(box.to_dict())
    np.testing.assert_allclose(restored.center, box.center)
    np.testing.assert_allclose(restored.size, box.size)
    assert restored.yaw == pytest.approx(box.yaw)


def test_negative_size_is_rejected() -> None:
    with pytest.raises(ValueError):
        OrientedBox(np.zeros(3), np.array([1.0, -1.0, 1.0]))


@pytest.mark.parametrize("angle", [0.0, 3.5, -7.0, math.pi])
def test_wrap_angle_range(angle: float) -> None:
    wrapped = wrap_angle(angle)
    assert -math.pi <= wrapped < math.pi
    assert math.cos(wrapped) == pytest.approx(math.cos(angle))
    assert math.sin(wrapped) == pytest.approx(math.sin(angle), abs=1e-12)


def test_min_area_rectangle_recovers_rotated_rectangle() -> None:
    rng = np.random.default_rng(0)
    local = rng.uniform([-1.5, -0.4], [1.5, 0.4], size=(2000, 2))
    yaw = 0.6
    c, s = math.cos(yaw), math.sin(yaw)
    points = local @ np.array([[c, -s], [s, c]]).T + [2.0, 1.0]
    center, size, angle = min_area_rectangle(points)
    np.testing.assert_allclose(center, [2.0, 1.0], atol=0.02)
    np.testing.assert_allclose(sorted(size), [0.8, 3.0], atol=0.02)
    # The rectangle orientation is only defined modulo pi/2.
    assert math.remainder(angle - yaw, math.pi / 2) == pytest.approx(0.0, abs=0.01)


def test_min_area_rectangle_handles_degenerate_input() -> None:
    center, size, _ = min_area_rectangle(np.array([[0.0, 0.0], [1.0, 0.0]]))
    np.testing.assert_allclose(center, [0.5, 0.0])
    np.testing.assert_allclose(size, [1.0, 0.0])


def test_fit_box_puts_long_side_on_local_x() -> None:
    rng = np.random.default_rng(1)
    truth = OrientedBox(np.array([0.5, -1.0, 0.45]), np.array([1.6, 0.7, 0.9]), -0.3)
    local = rng.uniform(-0.5, 0.5, size=(3000, 3)) * truth.size
    c, s = math.cos(truth.yaw), math.sin(truth.yaw)
    xy = local[:, :2] @ np.array([[c, -s], [s, c]]).T
    points = np.column_stack([xy, local[:, 2]]) + truth.center
    fitted = fit_box(points)
    np.testing.assert_allclose(fitted.size, truth.size, atol=0.03)
    np.testing.assert_allclose(fitted.center, truth.center, atol=0.03)
    assert math.remainder(fitted.yaw - truth.yaw, math.pi) == pytest.approx(0.0, abs=0.02)


def test_fit_box_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        fit_box(np.zeros((0, 3)))


def test_trimmed_fit_handles_tiny_clouds() -> None:
    for count in (1, 2):
        fitted = fit_box(np.arange(3.0 * count).reshape(count, 3), trim=0.02)
        assert np.all(np.isfinite(fitted.size))
