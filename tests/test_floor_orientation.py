"""A dense low wall must not displace the actual horizontal floor candidate."""
import numpy as np
import pytest

from pivot.lifting.floor import detect_floor


def test_dominant_low_wall_does_not_win_over_a_visible_floor():
    rng = np.random.default_rng(12)
    floor = np.column_stack((rng.uniform(0, 4, (500, 2)), np.zeros(500)))
    wall = np.column_stack((np.zeros(6000), rng.uniform(0, 4, 6000),
                            rng.uniform(.04, 2.5, 6000)))
    points = np.vstack((floor, wall))
    # Rotate the entire input: no assumption that the reconstruction is z-up.
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    up = rotation @ np.array([0., 0., 1.])
    plane, _ = detect_floor(points @ rotation.T, up)
    assert abs(plane.normal @ up) > .99
    assert abs(plane.offset) < .02


def test_wall_only_input_is_not_accepted_as_a_floor():
    rng = np.random.default_rng(12)
    wall = np.column_stack((np.zeros(1000), rng.uniform(0, 4, 1000),
                            rng.uniform(0, 2.5, 1000)))
    with pytest.raises(ValueError, match="No plane"):
        detect_floor(wall, np.array([0., 0., 1.]))
