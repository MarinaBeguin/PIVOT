"""Tests of dropping objects bigger or smaller than any real object of their label."""

import numpy as np
import pytest

from pivot.geometry.box import OrientedBox, fit_box
from pivot.geometry.shape import VoxelShape
from pivot.geometry.types import FloatArray
from pivot.lifting.plausible import drop_hollow_tops, drop_implausible, separate_pieces
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject

RES = 0.04


def solid(lo: tuple[float, float, float], hi: tuple[float, float, float]) -> FloatArray:
    """Cell centers filling a block."""
    axes = [np.arange(a + RES / 2, b, RES) for a, b in zip(lo, hi)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)


def carved(object_id: int, label: str, points: FloatArray) -> SceneObject:
    """An object whose volume is the cells of some points, in a box around them."""
    fitted = fit_box(points, trim=0.0)
    around = box(tuple(fitted.center), tuple(fitted.size + RES), fitted.yaw)
    return SceneObject(object_id, label, around,
                       shape=VoxelShape.from_world_points(around, points, RES))


def fit(points: FloatArray) -> OrientedBox:
    return fit_box(points, trim=0.0)


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


def test_fragments_of_other_furniture_leave_a_table() -> None:
    top = solid((0.0, 0.0, 0.7), (1.4, 0.8, 0.76))
    stray = solid((0.2, 2.0, 0.6), (0.5, 2.3, 0.7))  # 1.2 m away: seen on another desk.
    table = carved(1, "table", np.vstack([top, stray]))
    assert max(table.box.size[:2]) > 2.0
    pieces = separate_pieces([table], fit)
    assert len(pieces) == 1
    kept = pieces[0]
    assert kept.object_id == 1
    assert sorted(kept.box.size[:2]) == pytest.approx((0.8, 1.4), abs=0.06)


def test_two_chairs_tracked_as_one_become_two() -> None:
    first = solid((0.0, 0.0, 0.0), (0.5, 0.5, 0.9))
    second = solid((1.5, 0.0, 0.0), (2.0, 0.5, 0.9))
    lamp = carved(7, "lamp", np.vstack([solid((3, 3, 0), (3.2, 3.2, 1.5)),
                                        solid((3.6, 3, 1.5), (3.8, 3.2, 1.7))]))
    pieces = separate_pieces([carved(4, "chair", np.vstack([first, second])), lamp], fit)
    chairs = sorted((o for o in pieces if o.label == "chair"), key=lambda o: o.box.center[0])
    assert [o.object_id for o in chairs] == [4, 8]
    assert all(o.box.size[0] == pytest.approx(0.5, abs=0.05) for o in chairs)
    assert next(o for o in pieces if o.label == "lamp") is lamp  # Too small to separate.


def test_a_table_of_a_few_edges_is_dropped_and_a_solid_block_kept() -> None:
    edges = np.vstack([solid((0.0, 0.0, 0.7), (1.6, 0.08, 0.78)),
                       solid((0.0, 0.0, 0.7), (0.08, 0.5, 0.78))])
    block = solid((3.0, 0.0, 0.7), (5.0, 1.4, 0.78))  # Two desks pushed together.
    kept = drop_hollow_tops([carved(1, "table", edges), carved(2, "table", block),
                             carved(3, "chair", edges)])
    assert [o.object_id for o in kept] == [2, 3]
