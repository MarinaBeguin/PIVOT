"""Tables in rows, set on the desk tops the video saw.

In a classroom or a meeting room, desks stand end to end in long rows. SAM 3.1
tracks a desk through hundreds of frames, and a track that drifts from one
desk to the next gives a table across the aisle, half on the chairs, while
stretches of a row are left with no table at all. The desk tops themselves
are plain in the scene's points: dense strips at the tables' height.

So when a room has many tables in rows (six tables and three rows at least),
the rows are found from those points (the peaks of their profile across the
rows), each table is set on the row it covers most, over the stretch where
the row is seen, and tables of a row are cut where they overlap. A stretch of
a row seen with no table becomes one: a surface at desk height was observed
there, and leaving it empty would tell the agent it can walk through. A table
over no row is left as it is.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy import ndimage

from pivot.geometry.box import OrientedBox
from pivot.geometry.floor import FloorMap
from pivot.geometry.shape import VoxelShape
from pivot.geometry.types import FloatArray
from pivot.scene.graph import SceneObject

# Labels of the furniture set in rows.
ROW_LABELS = frozenset({"table", "desk"})
# Rows are looked for with at least this many tables, and kept when there are this
# many: a classroom or a meeting room, not a home's few tables.
MIN_TABLES = 6
MIN_ROWS = 3
# Points this far below and above the tables' median top are desk tops, in meters.
TOP_BELOW, TOP_ABOVE = 0.08, 0.03
# Points closer than this to the floor's edge are walls, in meters.
WALL_MARGIN = 0.3
# Profile bin, in meters, and the share of its high values a row's bins reach.
BIN = 0.04
ROW_LEVEL = 0.35
# A row is then narrowed to where its points reach this share of its own peak: the
# desk top, without the edges of the chairs pushed under it.
TOP_LEVEL = 0.6
# A row is this wide at least and at most, across, in meters.
ROW_WIDTH = (0.3, 1.6)
# Along a row, a stretch has at least this share of the row's typical density.
DENSE_SHARE = 0.3
# Gaps shorter than this many bins do not cut a stretch.
GAP_BINS = 3
# Shortest table kept after cutting, and shortest new one, in meters.
MIN_LENGTH = 0.3
MIN_NEW_LENGTH = 0.6
# Volume cells of the tables set on the rows, in meters.
RESOLUTION = 0.04


@dataclass(frozen=True)
class Rows:
    """Rows of desks, in the room's frame.

    Attributes:
        yaw: Orientation of the room's walls, in radians.
        axis: 0 if the rows run along the room's first axis, else 1.
        spans: Interval of each row across the rows, in meters.
        height: The tables' top, in meters.
    """

    yaw: float
    axis: int
    spans: list[tuple[float, float]]
    height: float

    def to_room(self, xy: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Gives points' coordinates along and across the rows.

        Args:
            xy: (N, 2) world points.

        Returns:
            Along and across, each (N,).
        """
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        u = xy[:, 0] * c + xy[:, 1] * s
        v = -xy[:, 0] * s + xy[:, 1] * c
        return (u, v) if self.axis == 0 else (v, u)

    def box(self, along: tuple[float, float], across: tuple[float, float]) -> OrientedBox:
        """Gives the box of a table on a stretch of a row.

        Args:
            along: Its interval along the row.
            across: The row's interval across.

        Returns:
            The box, from the floor to the tables' top.
        """
        a, b = 0.5 * (along[0] + along[1]), 0.5 * (across[0] + across[1])
        u, v = (a, b) if self.axis == 0 else (b, a)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        center = np.array([u * c - v * s, u * s + v * c, 0.5 * self.height])
        length, width = along[1] - along[0], across[1] - across[0]
        size = [length, width] if self.axis == 0 else [width, length]
        return OrientedBox(center, np.array([*size, self.height]), self.yaw)


def _desk_tops(points: FloatArray, height: float, floor_xy: FloatArray) -> FloatArray:
    """Keeps the points at the tables' height, away from the walls.

    Args:
        points: (N, 3) leveled scene points.
        height: The tables' top.
        floor_xy: (M, 2) floor points, for the walls.

    Returns:
        (K, 2) xy of the desk-top points.
    """
    band = points[(points[:, 2] > height - TOP_BELOW) & (points[:, 2] < height + TOP_ABOVE)]
    floor = FloorMap.from_points(floor_xy)
    inner = ndimage.distance_transform_edt(floor.floor) * floor.resolution
    col = np.floor((band[:, 0] - floor.origin[0]) / floor.resolution).astype(int)
    row = np.floor((band[:, 1] - floor.origin[1]) / floor.resolution).astype(int)
    inside = (row >= 0) & (row < inner.shape[0]) & (col >= 0) & (col < inner.shape[1])
    away = np.zeros(len(band), dtype=bool)
    away[inside] = inner[row[inside], col[inside]] > WALL_MARGIN
    result: FloatArray = band[away, :2]
    return result


def _runs(values: FloatArray, start: float, stop: float, level: float,
          min_length: float) -> list[tuple[float, float]]:
    """Finds the stretches where values are dense.

    Args:
        values: (N,) positions.
        start: Where to look from.
        stop: Where to look to.
        level: Fewest positions in a bin of a stretch.
        min_length: Shortest stretch kept.

    Returns:
        The stretches, in order.
    """
    edges = np.arange(start, stop + BIN, BIN)
    if len(edges) < 2:
        return []
    dense = ndimage.binary_closing(np.histogram(values, edges)[0] >= level, iterations=GAP_BINS)
    labels, count = ndimage.label(dense)
    runs = []
    for index in range(1, count + 1):
        bins = np.nonzero(labels == index)[0]
        a, b = float(edges[bins[0]]), float(edges[bins[-1] + 1])
        if b - a >= min_length:
            runs.append((a, b))
    return runs


def _row_spans(across: FloatArray) -> tuple[float, list[tuple[float, float]]]:
    """Finds the rows in the profile of desk-top points across them.

    Args:
        across: (N,) positions of the desk-top points across the rows.

    Returns:
        How sharp the profile is (its standard deviation over its mean), and
        the interval of each row.
    """
    edges = np.arange(across.min(), across.max() + BIN, BIN)
    raw = np.histogram(across, edges)[0].astype(float)
    profile = ndimage.uniform_filter1d(raw, 3)
    labels, count = ndimage.label(profile > ROW_LEVEL * np.percentile(profile, 90))
    spans = []
    for index in range(1, count + 1):
        bins = np.nonzero(labels == index)[0]
        top = bins[raw[bins] >= TOP_LEVEL * raw[bins].max()]
        a, b = float(edges[top[0]]), float(edges[top[-1] + 1])
        if ROW_WIDTH[0] <= b - a <= ROW_WIDTH[1]:
            spans.append((a, b))
    return float(profile.std() / (profile.mean() + 1e-9)), spans


def find_rows(objects: list[SceneObject], points: FloatArray, floor_xy: FloatArray,
              yaw: float) -> Rows | None:
    """Finds rows of desks from the scene's points.

    Args:
        objects: The objects.
        points: (N, 3) leveled scene points.
        floor_xy: (M, 2) floor points.
        yaw: Orientation of the room's walls, in radians.

    Returns:
        The rows; None with few tables or no clear rows.
    """
    tables = [o for o in objects if o.label.lower() in ROW_LABELS]
    if len(tables) < MIN_TABLES or len(floor_xy) == 0:
        return None
    height = float(np.median([o.box.z_max for o in tables]))
    tops = _desk_tops(points, height, floor_xy)
    if len(tops) == 0:
        return None
    found = []
    for axis in (0, 1):
        contrast, spans = _row_spans(Rows(yaw, axis, [], height).to_room(tops)[1])
        if len(spans) >= MIN_ROWS:
            found.append((contrast, axis, spans))
    if not found:
        return None
    _, axis, spans = max(found, key=lambda f: f[0])
    return Rows(yaw, axis, spans, height)


def _volume(box: OrientedBox, points: FloatArray) -> VoxelShape:
    """Gives a table on a row the volume its box holds of the scene's points.

    Args:
        box: The table's box.
        points: (N, 3) leveled scene points.

    Returns:
        The occupied cells (the whole box if it holds no point).
    """
    local = box.to_local(points)
    inside = points[np.all(np.abs(local) <= 0.5 * box.size, axis=1) & (points[:, 2] > 0.03)]
    if len(inside) == 0:
        inside = np.array([box.center])
    return VoxelShape.from_world_points(box, inside, RESOLUTION)


@dataclass
class _Row:
    """One row while its tables are set: its desk-top points and its tables.

    Attributes:
        span: Its interval across the rows.
        along: Positions of its desk-top points along it.
        dense: Fewest points in a bin of a stretch where a desk top is seen.
        tables: ``(id, start, end)`` of the tables set on it, along it.
    """

    span: tuple[float, float]
    along: FloatArray
    dense: float
    tables: list[tuple[int, float, float]]


def _rows_of_points(rows: Rows, points: FloatArray, floor_xy: FloatArray) -> list[_Row]:
    """Gives each row its desk-top points and its density.

    Args:
        rows: The rows.
        points: (N, 3) leveled scene points.
        floor_xy: (M, 2) floor points.

    Returns:
        The rows, no table set yet.
    """
    along, across = rows.to_room(_desk_tops(points, rows.height, floor_xy))
    edges = np.arange(along.min(), along.max() + BIN, BIN)
    result = []
    for a, b in rows.spans:
        mine = along[(across >= a) & (across <= b)]
        counts = np.histogram(mine, edges)[0]
        dense = max(4.0, DENSE_SHARE * float(np.median(counts[counts > 0]))) if len(mine) else 4.0
        result.append(_Row((a, b), mine, dense, []))
    return result


def _set_on_a_row(obj: SceneObject, rows: Rows, on_rows: list[_Row]) -> bool:
    """Sets a table on the row it covers most, over the stretch where the row is seen.

    Args:
        obj: The table.
        rows: The rows.
        on_rows: The rows' points; the table is added to its row's.

    Returns:
        False if it covers no row seen there.
    """
    f_along, f_across = rows.to_room(obj.box.footprint())
    lo, hi = float(f_along.min()), float(f_along.max())
    overlap = [min(r.span[1], float(f_across.max())) - max(r.span[0], float(f_across.min()))
               for r in on_rows]
    row = on_rows[int(np.argmax(overlap))]
    seen = _runs(row.along, lo - 0.1, hi + 0.1, row.dense, MIN_LENGTH)
    if max(overlap) <= 0 or not seen:
        return False
    row.tables.append((obj.object_id, seen[0][0], seen[-1][1]))
    return True


def _lay_row(row: _Row, rows: Rows, by_id: dict[int, SceneObject], points: FloatArray,
             next_id: int) -> list[SceneObject]:
    """Lays a row's tables end to end, and fills the stretches seen with none.

    Args:
        row: The row and its tables.
        rows: The rows.
        by_id: The objects, by id.
        points: (N, 3) leveled scene points, for the tables' volumes.
        next_id: Id of the first new table.

    Returns:
        The row's tables, the new ones last.
    """
    tables = sorted(row.tables, key=lambda t: t[1])
    stretches = [[t[1], t[2]] for t in tables]
    # Tables of one row that overlap meet halfway through their overlap.
    for k in range(len(stretches) - 1):
        if stretches[k][1] > stretches[k + 1][0]:
            stretches[k][1] = stretches[k + 1][0] = 0.5 * (stretches[k][1] + stretches[k + 1][0])
    laid, covered = [], []
    for (object_id, _, _), (lo, hi) in zip(tables, stretches):
        if hi - lo >= MIN_LENGTH:
            box = rows.box((lo, hi), row.span)
            laid.append(replace(by_id[object_id], box=box, shape=_volume(box, points)))
            covered.append((lo, hi))
    for lo, hi in _uncovered(row, covered):
        box = rows.box((lo, hi), row.span)
        laid.append(SceneObject(next_id, "table", box, movable=True, confidence=0.5,
                                shape=_volume(box, points)))
        next_id += 1
    return laid


def _uncovered(row: _Row, covered: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Finds the stretches of a row where a desk top was seen and no table stands.

    Args:
        row: The row.
        covered: Stretches of its tables, along it.

    Returns:
        The stretches long enough for a table.
    """
    if len(row.along) == 0:
        return []
    free = _runs(row.along, float(row.along.min()), float(row.along.max()), row.dense,
                 MIN_NEW_LENGTH)
    for c_lo, c_hi in covered:
        free = [part for p_lo, p_hi in free
                for part in ((p_lo, min(p_hi, c_lo)), (max(p_lo, c_hi), p_hi))
                if part[1] - part[0] >= MIN_NEW_LENGTH]
    return free


def set_tables_on_rows(objects: list[SceneObject], points: FloatArray,
                       floor_xy: FloatArray, yaw: float) -> list[SceneObject]:
    """Sets tables on the rows of desk tops the video saw, and fills the rows' gaps.

    Args:
        objects: The objects, volumes carved.
        points: (N, 3) leveled scene points.
        floor_xy: (M, 2) floor points.
        yaw: Orientation of the room's walls, in radians.

    Returns:
        The objects; new tables get ids after the largest one.
    """
    rows = find_rows(objects, points, floor_xy, yaw)
    if rows is None:
        return objects
    on_rows = _rows_of_points(rows, points, floor_xy)
    kept = [o for o in objects if o.label.lower() not in ROW_LABELS
            or not _set_on_a_row(o, rows, on_rows)]
    by_id = {o.object_id: o for o in objects}
    next_id = max(by_id, default=0) + 1
    for row in on_rows:
        laid = _lay_row(row, rows, by_id, points, next_id)
        next_id = max([next_id - 1, *(o.object_id for o in laid)]) + 1
        kept.extend(laid)
    return kept
