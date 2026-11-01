"""Which object each point of the cloud belongs to, so that it moves with it.

The viewer draws the reconstructed points. When an object is moved or taken
out of the room, its points must go with it: left behind, they show a ghost of
the old sofa where the new one should be. A point belongs to a movable object
when it lies in one of the object's occupied cells (or next to one), inside its
box; the floor under the object stays. Where boxes overlap (a lamp on a
table), the smaller object wins.
"""

from __future__ import annotations

import itertools

import numpy as np

from pivot.geometry.types import FloatArray, IntArray
from pivot.scene.graph import Scene, SceneObject, is_flat, is_opening

# Points this close outside a box, or to an occupied cell, still belong to it, in meters.
_MARGIN = 0.03
# Points this close above the bottom of an object standing on the floor are floor.
_FLOOR_BAND = 0.03
# Every cell next to a cell, itself included.
_NEIGHBORS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))


def point_owners(scene: Scene, points: FloatArray) -> IntArray:
    """Finds the movable object each point belongs to.

    Args:
        scene: The scene.
        points: (N, 3) points of the cloud.

    Returns:
        (N,) object ids, -1 for points of no movable object (walls, floor,
        fixed furniture).
    """
    owners = np.full(len(points), -1, dtype=np.int64)
    movable = [o for o in scene.objects.values() if o.movable and not is_opening(o)]
    # Larger objects first: a smaller one inside or on top of them takes its points back.
    for obj in sorted(movable, key=lambda o: -float(np.prod(o.box.size))):
        owners[_points_of(obj, points)] = obj.object_id
    return owners


def _points_of(obj: SceneObject, points: FloatArray) -> IntArray:
    """Lists the points of one object.

    Args:
        obj: The object.
        points: (N, 3) points of the cloud.

    Returns:
        Indices of the points in its box and its occupied cells.
    """
    box = obj.box
    reach = 0.5 * float(np.hypot(*box.size[:2])) + _MARGIN
    near = np.flatnonzero((np.abs(points[:, 0] - box.center[0]) <= reach)
                          & (np.abs(points[:, 1] - box.center[1]) <= reach)
                          & (points[:, 2] >= box.z_min - _MARGIN)
                          & (points[:, 2] <= box.z_max + _MARGIN))
    if len(near) == 0:
        return near
    local = box.to_local(points[near])
    inside = np.all(np.abs(local) <= 0.5 * box.size + _MARGIN, axis=1)
    if not is_flat(obj) and box.z_min < 0.05:
        inside &= points[near, 2] > box.z_min + _FLOOR_BAND
    if obj.shape is not None:
        cells = np.floor(local / obj.shape.resolution).astype(np.int64)
        occupied = np.zeros(len(near), dtype=bool)
        for offset in _NEIGHBORS:
            occupied |= obj.shape.contains_cells(cells + offset)
        inside &= occupied
    result: IntArray = near[inside]
    return result
