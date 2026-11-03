"""Tests of setting tables on the rows of desk tops the video saw."""

import numpy as np

from pivot.lifting.rows import find_rows, set_tables_on_rows
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject

# Four rows of desks, 60 cm wide, from y = 1 to 6 m, in a room of 8 x 7 m.
ROWS = [(1.0, 1.6), (2.6, 3.2), (4.2, 4.8), (5.8, 6.4)]


def classroom_points() -> np.ndarray:
    rng = np.random.default_rng(3)
    floor = np.column_stack([rng.uniform(0, 8, 60000), rng.uniform(0, 7, 60000), np.zeros(60000)])
    tops = [np.column_stack([rng.uniform(a, b, 8000), rng.uniform(1.0, 6.0, 8000),
                             rng.uniform(0.73, 0.77, 8000)]) for a, b in ROWS]
    return np.vstack([floor, *tops])


def table(object_id: int, x: tuple[float, float], y: tuple[float, float]) -> SceneObject:
    return SceneObject(object_id, "table", box((0.5 * (x[0] + x[1]), 0.5 * (y[0] + y[1]), 0.375),
                                              (x[1] - x[0], y[1] - y[0], 0.75)))


def test_rows_are_found_from_the_desk_tops() -> None:
    points = classroom_points()
    tables = [table(i, ROWS[i % 4], (1.0, 2.5)) for i in range(6)]
    rows = find_rows(tables, points, points[points[:, 2] < 0.05, :2], 0.0)
    assert rows is not None and rows.axis == 1 and len(rows.spans) == 4
    for (a, b), (ra, rb) in zip(ROWS, rows.spans):
        assert abs(a - ra) < 0.1 and abs(b - rb) < 0.1


def test_a_table_across_an_aisle_is_set_on_its_row_and_gaps_get_tables() -> None:
    points = classroom_points()
    tables = [table(1, (1.3, 2.3), (1.0, 2.5)),  # Across the aisle between rows 1 and 2.
              table(2, (2.6, 3.2), (1.0, 3.5)), table(3, (2.6, 3.2), (3.0, 6.0)),  # Overlap.
              table(4, (4.2, 4.8), (1.0, 6.0)), table(5, (5.8, 6.4), (1.0, 6.0)),
              table(6, (5.8, 6.4), (1.0, 6.0))]
    chair = SceneObject(7, "chair", box((2.0, 4.0, 0.45), (0.5, 0.5, 0.9)))
    result = {o.object_id: o for o in set_tables_on_rows(tables + [chair], points,
                                                        points[points[:, 2] < 0.05, :2], 0.0)}
    first = result[1].box
    assert abs(first.center[0] - 1.3) < 0.1 and first.size[0] < 0.75  # On row 1, not the aisle.
    # The overlapping tables of row 2 meet halfway.
    assert result[2].box.center[1] + 0.5 * result[2].box.size[1] <= (
        result[3].box.center[1] - 0.5 * result[3].box.size[1] + 1e-6)
    # Row 1 is seen from y = 1 to 6 m, its table covers 1 to 2.5: the rest is a new table.
    new = [o for i, o in result.items() if i > 7]
    assert len(new) == 1 and new[0].label == "table" and abs(new[0].box.center[0] - 1.3) < 0.1
    assert result[7] is chair


def test_a_home_with_a_few_tables_is_left_alone() -> None:
    points = classroom_points()
    tables = [table(1, (1.3, 2.3), (1.0, 2.5)), table(2, (2.6, 3.2), (1.0, 3.5))]
    assert set_tables_on_rows(tables, points, points[points[:, 2] < 0.05, :2], 0.0) == tables
