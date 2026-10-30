"""Shared fixtures."""

import pytest

from pivot.scene.examples import box, office_scene
from pivot.scene.graph import Scene

__all__ = ["box"]


@pytest.fixture(name="office")
def fixture_office() -> Scene:
    """The synthetic 5 x 4 m office of :func:`pivot.scene.examples.office_scene`."""
    return office_scene()
