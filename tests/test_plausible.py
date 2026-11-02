"""Tests of dropping objects bigger or smaller than any real object of their label."""

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


def test_objects_smaller_than_their_label_allows_are_dropped() -> None:
    bed = SceneObject(1, "bed", box((2.0, 2.0, 0.31), (1.5, 1.0, 0.62)))
    sliver = SceneObject(2, "bed", box((2.0, 2.6, 0.26), (1.29, 0.27, 0.52)))
    low_desk = SceneObject(3, "desk", box((1.0, 3.0, 0.25), (0.62, 0.32, 0.5)))
    tall_monitor = SceneObject(4, "monitor", box((3.0, 1.0, 0.55), (0.83, 0.18, 1.1)))
    row_of_tables = SceneObject(5, "table", box((5.0, 4.0, 0.4), (4.5, 0.76, 0.8)))
    kept = drop_implausible([bed, sliver, low_desk, tall_monitor, row_of_tables])
    assert [o.object_id for o in kept] == [1, 5]
