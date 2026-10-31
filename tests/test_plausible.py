"""Tests of dropping objects bigger than any real object of their label."""

from pivot.lifting.plausible import drop_implausible
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject


def test_objects_bigger_than_their_label_allows_are_dropped() -> None:
    dresser = SceneObject(1, "dresser", box((1.0, 1.0, 0.5), (1.2, 0.5, 1.0)))
    leak = SceneObject(2, "dresser", box((3.0, 3.0, 1.25), (2.9, 2.9, 2.5)))
    thick_rug = SceneObject(3, "rug", box((2.0, 1.0, 0.23), (0.6, 0.5, 0.46)))
    statue = SceneObject(4, "statue", box((4.0, 1.0, 1.5), (3.0, 3.0, 3.0)))  # No known limit.
    kept = drop_implausible([dresser, leak, thick_rug, statue])
    assert [o.object_id for o in kept] == [1, 4]
