"""Shared fixtures: a small synthetic office."""

import numpy as np
import pytest

from pivot.geometry.box import OrientedBox
from pivot.geometry.floor import FloorMap
from pivot.scene.graph import Scene, SceneObject


def box(center: tuple[float, float, float], size: tuple[float, float, float],
        yaw: float = 0.0) -> OrientedBox:
    """Shorthand for building a box from tuples."""
    return OrientedBox(np.array(center), np.array(size), yaw)


@pytest.fixture(name="office")
def fixture_office() -> Scene:
    """A 5 x 4 m office: desk with a lamp on it, chair, shelf and a window.

    Layout (top view, meters)::

        y=4 +-------------[window]---------+
            |                    shelf     |
            |   desk+lamp                  |
            |   chair                      |
        y=0 +------------------------------+
           x=0                            x=5
    """
    xs, ys = np.meshgrid(np.arange(0.0, 5.0, 0.02), np.arange(0.0, 4.0, 0.02))
    floor = FloorMap.from_points(np.column_stack([xs.ravel(), ys.ravel()]), close_radius=0)
    objects = [
        SceneObject(1, "desk", box((1.5, 2.5, 0.375), (1.4, 0.7, 0.75))),
        SceneObject(2, "lamp", box((1.2, 2.6, 0.95), (0.2, 0.2, 0.4))),
        SceneObject(3, "office chair", box((1.5, 1.7, 0.5), (0.6, 0.6, 1.0))),
        SceneObject(4, "shelf", box((4.0, 3.8, 1.0), (1.0, 0.4, 2.0))),
        SceneObject(5, "window", box((2.8, 3.95, 1.6), (1.2, 0.1, 1.2)), movable=False),
    ]
    return Scene.from_objects(objects, floor)
