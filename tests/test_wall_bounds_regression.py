"""Wall correction must not invert thin or wholly external detections."""

import numpy as np
import pytest

from pivot.lifting.agreement import _within_walls
from pivot.lifting.openings import Walls
from pivot.scene.examples import box


@pytest.mark.parametrize("yaw", [0.0, 0.04, np.pi / 2, np.pi + 0.03])
@pytest.mark.parametrize("center", [(2.0, -0.15, 0.5), (-0.15, 2.0, 0.5)])
def test_wholly_external_thin_box_does_not_invert(yaw, center):
    original = box(center, (1.11243906, 0.03, 0.03), yaw)
    walls = Walls(0.0, np.array([0., 0.]), np.array([4., 4.]))
    corrected = _within_walls(original, walls)
    assert np.all(np.isfinite(corrected.size))
    assert np.all(corrected.size > 0)


def test_valid_partial_intersection_is_still_trimmed():
    original = box((2., 0.15, 0.5), (1., 0.5, 1.))
    walls = Walls(0.0, np.array([0., 0.]), np.array([4., 4.]))
    corrected = _within_walls(original, walls)
    assert corrected.size[1] == pytest.approx(0.4)
    assert corrected.center[1] == pytest.approx(0.2)
