"""Dropping objects bigger or smaller than any real object of their label.

A mask that leaks onto a wall or the floor, or that follows a reflection,
lifts to a box meters wide: a dresser of 3 x 3 x 2.5 m, a rug half a meter
thick. No object of the label is that big, so the object is a false
detection and is dropped rather than left to block the room. A mask on a
small part of a big piece of furniture, or on something else, lifts to a
box too small for its label: a "bed" 27 cm wide along the real bed, a
"desk" 50 cm high that is a bin. It is dropped too. The limits are
generous: real furniture stays well inside them.

Once the volumes are carved, two more rules hold, so that the room shows
only furniture that is there:

- An object is one piece. A track that mixed in points of other furniture
  (a classroom's rows of desks, tracked in pieces of video) gives a table
  with fragments up to a meter and a half away, which stretch its box. The
  fragments are taken out; a piece big enough to be furniture of its own (two
  chairs tracked as one) becomes a separate object.
- A table's top is a slab. A "table" of a few edges seen at a grazing angle,
  that fills less than half of its box and is smaller than any real table,
  is dropped. A solid block is kept whole, even when it is likely two desks
  pushed together: splitting it without a gap to cut along would be a guess.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
from scipy import ndimage

from pivot.geometry.box import OrientedBox
from pivot.geometry.shape import VoxelShape
from pivot.geometry.types import BoolArray, FloatArray, IntArray
from pivot.scene.graph import (
    SURFACE_LABELS, SceneObject, is_flat, is_opening, is_wall_mounted,
)

# Largest size of one object of a label: long side, short side and height, in meters.
# Tables joined in a row (a classroom, a meeting room) are seen as one long table.
MAX_SIZE = {
    "chair": (1.5, 1.2, 1.5), "office chair": (1.0, 1.0, 1.5), "armchair": (1.4, 1.2, 1.3),
    "stool": (0.7, 0.7, 1.1), "sofa": (3.6, 2.2, 1.3), "bed": (2.6, 2.4, 1.8),
    "table": (6.0, 1.8, 1.2), "coffee table": (1.8, 1.2, 0.7), "desk": (2.6, 1.4, 1.3),
    "dresser": (2.2, 0.9, 1.8), "nightstand": (0.9, 0.8, 1.0), "tv stand": (2.6, 0.9, 1.1),
    "cabinet": (2.6, 1.0, 2.5), "wardrobe": (2.8, 1.0, 2.7), "shelf": (2.8, 0.9, 2.7),
    "bookshelf": (2.8, 0.9, 2.7), "television": (2.2, 0.5, 1.5), "monitor": (1.2, 0.5, 0.9),
    "trash can": (0.7, 0.7, 1.1), "rug": (4.5, 3.5, 0.15), "plant": (1.6, 1.6, 2.6),
}

# Smallest size of one object of a label, for big furniture: long side, short side
# and height, in meters.
MIN_SIZE = {
    "bed": (1.2, 0.7, 0.2), "sofa": (1.0, 0.6, 0.35), "desk": (0.6, 0.4, 0.6),
    "table": (0.4, 0.3, 0.3), "wardrobe": (0.6, 0.35, 1.2), "dresser": (0.5, 0.3, 0.5),
}


def plausible(obj: SceneObject) -> bool:
    """Tests whether an object is neither bigger nor smaller than real objects of its label.

    Args:
        obj: The object.

    Returns:
        True if its box is within the limits of its label (those it has).
    """
    long_side, short_side = sorted(obj.box.size[:2], reverse=True)
    size = (long_side, short_side, float(obj.box.size[2]))
    largest = MAX_SIZE.get(obj.label.lower())
    smallest = MIN_SIZE.get(obj.label.lower())
    return bool((largest is None or all(s <= m for s, m in zip(size, largest)))
                and (smallest is None or all(s >= m for s, m in zip(size, smallest))))


def drop_implausible(objects: list[SceneObject]) -> list[SceneObject]:
    """Drops objects bigger or smaller than any real object of their label.

    Args:
        objects: Objects with settled boxes.

    Returns:
        The plausible objects.
    """
    return [o for o in objects if plausible(o)]


# Cells this many cells apart are in one piece.
PIECE_GAP = 1
# Only objects whose main piece covers this much floor are separated, in square
# meters: a lamp's foot and head stay together.
PIECES_MIN_AREA = 0.15
# A piece at least this share of the main one, and this big, is an object of its
# own; a smaller one is a fragment and is dropped.
OWN_OBJECT_SHARE = 0.25
OWN_OBJECT_AREA = 0.1
# Fragments covering less than this share of the main piece are left in place: a
# few stray cells do not stretch a box.
FRAGMENTS_MIN_SHARE = 0.05
# A table smaller than this, in square meters, that fills less than this share
# of its box, is not a table top.
SLAB_AREA = 0.4
SLAB_FILL = 0.5

# Fits a box to points, aligned with the walls (see scene_builder.fit_object_box).
Fit = Callable[[FloatArray], OrientedBox]


def _footprint(points: FloatArray, resolution: float) -> tuple[BoolArray, IntArray]:
    """Rasterizes points seen from above.

    Args:
        points: (N, 3) world points.
        resolution: Cell side, in meters.

    Returns:
        The occupied cells as a grid, and each point's cell in it.
    """
    cells = np.floor(points[:, :2] / resolution).astype(np.int64)
    cells -= cells.min(axis=0)
    grid = np.zeros(tuple(cells.max(axis=0) + 1), dtype=bool)
    grid[cells[:, 0], cells[:, 1]] = True
    return grid, cells


def _refit(obj: SceneObject, points: FloatArray, fit: Fit, object_id: int) -> SceneObject:
    """Gives an object the box and volume of some of its cells.

    Args:
        obj: The object.
        points: (N, 3) centers of the cells kept.
        fit: Fits a box to points.
        object_id: Id of the result.

    Returns:
        The object, with a box around those cells and those cells as its volume.
    """
    assert obj.shape is not None
    resolution = obj.shape.resolution
    fitted = fit(points)
    # The box goes around the cells, not their centers.
    box = OrientedBox(fitted.center, fitted.size + resolution, fitted.yaw)
    return replace(obj, object_id=object_id, box=box,
                   shape=VoxelShape.from_world_points(box, points, resolution))


def separate_pieces(objects: list[SceneObject], fit: Fit) -> list[SceneObject]:
    """Makes every object one piece: fragments go, a big piece becomes its own object.

    Args:
        objects: Objects with carved volumes.
        fit: Fits a box to points, aligned with the walls.

    Returns:
        The objects; new ones get ids after the largest one.
    """
    next_id = max((o.object_id for o in objects), default=0) + 1
    result = []
    for obj in objects:
        if obj.shape is None or is_opening(obj) or is_wall_mounted(obj) or is_flat(obj):
            result.append(obj)
            continue
        points = obj.shape.world_centers(obj.box)
        piece, areas = _pieces(points, obj.shape.resolution)
        main = int(np.argmax(areas))
        rest = float(areas.sum() - areas[main])
        if areas[main] < PIECES_MIN_AREA or rest < FRAGMENTS_MIN_SHARE * areas[main]:
            result.append(obj)
            continue
        result.append(_refit(obj, points[piece == main], fit, obj.object_id))
        for index in range(1, len(areas)):
            if index != main and areas[index] >= max(OWN_OBJECT_AREA,
                                                     OWN_OBJECT_SHARE * areas[main]):
                result.append(_refit(obj, points[piece == index], fit, next_id))
                next_id += 1
    return result


def _pieces(points: FloatArray, resolution: float) -> tuple[IntArray, FloatArray]:
    """Splits an object's cells into the pieces they form, seen from above.

    Args:
        points: (N, 3) centers of its cells.
        resolution: Cell side, in meters.

    Returns:
        The piece of each cell (from 1), and the floor each piece covers, in
        square meters (index 0, the empty floor between them, is 0).
    """
    grid, cells = _footprint(points, resolution)
    labels, count = ndimage.label(ndimage.binary_dilation(grid, iterations=PIECE_GAP))
    areas: FloatArray = np.bincount(labels[grid], minlength=count + 1) * resolution**2
    areas[0] = 0.0
    piece: IntArray = labels[cells[:, 0], cells[:, 1]]
    return piece, areas


def is_slab(obj: SceneObject) -> bool:
    """Tests whether a table-like object's volume looks like a top, seen from above.

    Args:
        obj: The object.

    Returns:
        False for a small "table" that fills less than half of its box.
    """
    if obj.shape is None or obj.label.lower() not in SURFACE_LABELS:
        return True
    grid, _ = _footprint(obj.shape.world_centers(obj.box), obj.shape.resolution)
    area = float(grid.sum()) * obj.shape.resolution**2
    fill = area / float(obj.box.size[0] * obj.box.size[1])
    return area >= SLAB_AREA or fill >= SLAB_FILL


def drop_hollow_tops(objects: list[SceneObject]) -> list[SceneObject]:
    """Drops "tables" that are only a few edges.

    Args:
        objects: Objects with carved volumes.

    Returns:
        The objects whose tops are slabs.
    """
    return [o for o in objects if is_slab(o)]
