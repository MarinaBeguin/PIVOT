"""How well each object agrees with what the camera saw, and boxes fitted to agree best.

An object's box is checked against the scene's points, which are the depth
of every frame put in 3D, and against the object's own points, those of its
SAM 3.1 masks. Two numbers say how faithful the box is:

- precision: of the points seen inside the box, the share that are the
  object's. A table set across an aisle holds the floor and the chairs of the
  aisle: its precision is low.
- recall: of the object's points, the share inside its box. A box that misses
  half of a sofa has a low recall.

Their harmonic mean is the object's agreement, from 0 to 1, kept in the
scene's metadata. Before the volumes are carved, each side of a box is moved
out or in, a few centimeters at a time, while the agreement grows: the box
ends where the camera says the object is. Two more things are settled
against the camera's points at the same time:

- orientation: a box turned away from the walls is set square to them when
  that explains the points nearly as well. A chair's or a lamp's footprint is
  close to a square or a disk, whose smallest box turns on a few points; a
  chair truly turned between two tables keeps its turn, since a box square to
  the walls would take in the tables.
- walls: a box past the room's outer walls is brought back in: what is seen
  past a wall (through glass, in a reflection) is not the furniture. Only the
  outer walls count, so furniture against a wall is never cut.

Doors, windows, objects on the walls and rugs keep their boxes, set by their
own rules.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree

from pivot.geometry.box import OrientedBox
from pivot.geometry.types import BoolArray, FloatArray
from pivot.lifting.openings import Walls
from pivot.scene.graph import SceneObject, is_flat, is_opening, is_wall_mounted

# A scene point within this distance of one of an object's points is the object's, in meters.
OWN_DISTANCE = 0.03
# Points below this height are the floor, in meters; never counted.
FLOOR = 0.05
# A side moves this far at a time, and at most this far from where it was, in meters.
STEP = 0.04
REACH = 0.3
# A refit is kept when it raises the agreement by at least this much.
MIN_GAIN = 0.03
# A refit never leaves out more than this share of the object's points more than its
# box did: it moves and trims empty or foreign margins, never cuts into the object
# (a cut would be free space where something stands).
MAX_RECALL_LOSS = 0.02
# Objects with fewer own points are left as they are.
MIN_POINTS = 50
# Most scene points looked at around one object, for speed.
MAX_POINTS = 200_000
# A box square to the walls is taken when its agreement is at most this much lower,
# and its footprint around the object's points at most this much larger: aligning a
# disk or a square seen askew costs nothing; aligning a chair truly turned adds the
# empty corners around it.
ALIGN_SLACK = 0.05
ALIGN_AREA = 1.25
# A box beyond the room's outer walls by more than this is brought in, in meters; one
# beyond them by more than a meter is left (something else is wrong with it).
WALL_TOLERANCE = 0.02
MAX_BEYOND = 1.0
# An object's points beyond these percentiles of their own extent are stray (mask edges
# seen on what lies behind): a box is not stretched to catch them.
CORE_PERCENTILES = (1.0, 99.0)


def _f1(local: FloatArray, own: BoolArray, sides: FloatArray,
        half: FloatArray) -> tuple[float, float]:
    """Measures the agreement of a box whose sides are moved.

    Args:
        local: (N, 3) scene points in the box's frame.
        own: (N,) which points are the object's.
        sides: Offsets of the box's sides: -x, +x, -y, +y, in meters (positive out).
        half: The box's half sizes.

    Returns:
        The harmonic mean of precision and recall, and the recall.
    """
    inside = ((local[:, 0] >= -half[0] - sides[0]) & (local[:, 0] <= half[0] + sides[1])
              & (local[:, 1] >= -half[1] - sides[2]) & (local[:, 1] <= half[1] + sides[3])
              & (np.abs(local[:, 2]) <= half[2]))
    total_own = int(own.sum())
    caught = int((inside & own).sum())
    held = int(inside.sum())
    if caught == 0 or held == 0 or total_own == 0:
        return 0.0, 0.0
    precision, recall = caught / held, caught / total_own
    return 2 * precision * recall / (precision + recall), recall


def _neighborhood(box: OrientedBox, scene: FloatArray,
                  tree: cKDTree) -> tuple[FloatArray, BoolArray]:
    """Takes the scene points around a box, in its frame, and marks the object's.

    Args:
        box: The object's box.
        scene: (N, 3) scene points above the floor.
        tree: Search tree of the object's own points.

    Returns:
        The points in the box's frame, and which are the object's.
    """
    local = box.to_local(scene)
    reach = 0.5 * box.size + REACH + STEP
    near = np.all(np.abs(local) <= reach, axis=1)
    local = local[near]
    if len(local) > MAX_POINTS:
        local = local[np.random.default_rng(0).choice(len(local), MAX_POINTS, replace=False)]
    distances, _ = tree.query(_to_world(box, local), distance_upper_bound=OWN_DISTANCE)
    own = np.isfinite(distances)
    # The object's extent, from its own points (not the scene points next to them):
    # beyond it, a scene point near one of its stray points is not the object's.
    lo, hi = np.percentile(box.to_local(tree.data)[:, :2], CORE_PERCENTILES, axis=0)
    own &= np.all((local[:, :2] >= lo) & (local[:, :2] <= hi), axis=1)
    return local, own


def _to_world(box: OrientedBox, local: FloatArray) -> FloatArray:
    """Puts points of a box's frame back in the world.

    Args:
        box: The box.
        local: (N, 3) points in its frame.

    Returns:
        (N, 3) world points.
    """
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    xy = local[:, :2] @ np.array([[c, s], [-s, c]]) + box.center[:2]
    result: FloatArray = np.column_stack([xy, local[:, 2] + box.center[2]])
    return result


def _refittable(obj: SceneObject) -> bool:
    """Tests whether an object's box is refit to agree with the camera.

    Args:
        obj: The object.

    Returns:
        False for openings, objects on the walls, rugs and fixed objects.
    """
    return obj.movable and not (is_opening(obj) or is_wall_mounted(obj) or is_flat(obj))


def _moved_box(box: OrientedBox, sides: FloatArray) -> OrientedBox:
    """Gives a box with its sides moved.

    Args:
        box: The box.
        sides: Offsets of its sides: -x, +x, -y, +y (positive out).

    Returns:
        The new box, same height and orientation.
    """
    shift = np.array([0.5 * (sides[1] - sides[0]), 0.5 * (sides[3] - sides[2])])
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    center = box.center.copy()
    center[:2] += np.array([[c, -s], [s, c]]) @ shift
    size = box.size.copy()
    size[0] += sides[0] + sides[1]
    size[1] += sides[2] + sides[3]
    return OrientedBox(center, size, box.yaw)


def _climb(local: FloatArray, own: BoolArray, half: FloatArray) -> tuple[FloatArray, float]:
    """Moves a box's sides one step at a time while its agreement grows.

    Args:
        local: (N, 3) scene points in the box's frame.
        own: (N,) which are the object's.
        half: The box's half sizes.

    Returns:
        The sides' offsets, and the agreement reached.
    """
    sides = np.zeros(4)
    best, recall = _f1(local, own, sides, half)
    floor = recall - MAX_RECALL_LOSS
    improved = True
    while improved:
        improved = False
        for side in range(4):
            for step in (STEP, -STEP):
                trial = sides.copy()
                trial[side] += step
                # A side stays within reach, and the box keeps some width.
                if abs(trial[side]) > REACH or half[side // 2] * 2 + trial[2 * (side // 2)] \
                        + trial[2 * (side // 2) + 1] < 2 * STEP:
                    continue
                score, kept = _f1(local, own, trial, half)
                if score > best + 1e-6 and kept >= floor:
                    sides, best, improved = trial, score, True
    return sides, best


def _core_area(own_points: FloatArray, box: OrientedBox) -> float:
    """Measures the footprint of an object's points, in a box's orientation.

    Args:
        own_points: (N, 3) its points.
        box: A box, for its orientation.

    Returns:
        The area of the core of the points' extent, in square meters.
    """
    lo, hi = np.percentile(box.to_local(own_points)[:, :2], CORE_PERCENTILES, axis=0)
    return float(np.prod(hi - lo + STEP))


def _square_box(box: OrientedBox, own_points: FloatArray, room_yaw: float) -> OrientedBox:
    """Gives the box of an object's points, square to the walls.

    Args:
        box: Its box.
        own_points: (N, 3) its points.
        room_yaw: Orientation of the room's walls, in radians.

    Returns:
        The box around the core of its points, turned to the nearest wall axis.
    """
    quarter = 0.5 * np.pi
    yaw = room_yaw + quarter * round((box.yaw - room_yaw) / quarter)
    turned = OrientedBox(box.center, box.size, yaw)
    lo, hi = np.percentile(turned.to_local(own_points)[:, :2], CORE_PERCENTILES, axis=0)
    mid = 0.5 * (lo + hi)
    c, s = np.cos(yaw), np.sin(yaw)
    center = box.center.copy()
    center[:2] += np.array([[c, -s], [s, c]]) @ mid
    return OrientedBox(center, np.array([hi[0] - lo[0] + STEP, hi[1] - lo[1] + STEP,
                                         box.size[2]]), yaw)


def _sides_in(turn: float, over_low: FloatArray, over_high: FloatArray) -> FloatArray:
    """Gives the side offsets that bring a box square to the walls back inside them.

    Args:
        turn: The box's orientation in the room's frame, in radians (near a quarter turn).
        over_low: (2,) how far it stands past the low walls, along the room's axes.
        over_high: (2,) how far past the high walls.

    Returns:
        Offsets of its sides -x, +x, -y, +y (negative in).
    """
    c, s = np.cos(turn), np.sin(turn)
    sides = np.zeros(4)
    for axis in range(2):
        # Box axis 0 runs along (c, s) in the room's frame, box axis 1 along (-s, c).
        along = np.array([c, s]) if axis == 0 else np.array([-s, c])
        room_axis = int(abs(along[1]) > abs(along[0]))
        low_side, high_side = ((2 * axis, 2 * axis + 1) if along[room_axis] > 0
                               else (2 * axis + 1, 2 * axis))
        sides[low_side] -= over_low[room_axis]
        sides[high_side] -= over_high[room_axis]
    return sides


def _within_walls(box: OrientedBox, walls: Walls) -> OrientedBox:
    """Brings a box that stands beyond the room's outer walls back in.

    Args:
        box: The box.
        walls: The room's walls.

    Returns:
        The box: its sides beyond a wall moved onto it (a box square to the
        walls), or the box moved in (a turned one); as it is past a meter.
    """
    corners = walls.to_room(box.footprint())
    over_low = np.maximum(walls.low - corners.min(axis=0), 0.0)
    over_high = np.maximum(corners.max(axis=0) - walls.high, 0.0)
    if max(over_low.max(), over_high.max()) <= WALL_TOLERANCE \
            or max(over_low.max(), over_high.max()) > MAX_BEYOND:
        return box
    turn = (box.yaw - walls.yaw) % (2 * np.pi)
    if min(abs(np.cos(turn)), abs(np.sin(turn))) < np.sin(np.radians(5.0)):
        sides = _sides_in(turn, over_low, over_high)
        remaining = box.size[:2] + np.array([sides[0] + sides[1], sides[2] + sides[3]])
        # A thin or wholly external detection may have no intersection with
        # the room. Clamping it to zero fabricates a surface on the wall.
        # Keep the observed box for scoring instead of creating invalid geometry.
        if np.any(remaining <= 1e-9):
            return box
        return _moved_box(box, sides)
    shift_room = over_low - over_high
    cw, sw = np.cos(walls.yaw), np.sin(walls.yaw)
    center = box.center.copy()
    center[:2] += np.array([[cw, -sw], [sw, cw]]) @ shift_room
    return OrientedBox(center, box.size, box.yaw)


def agree_with_camera(objects: list[SceneObject], object_points: dict[int, FloatArray],
                      scene_points: FloatArray, refit: bool = True,
                      room_yaw: float | None = None, walls: Walls | None = None,
                      ) -> tuple[list[SceneObject], dict[int, float]]:
    """Refits boxes to agree with the camera, and scores every object's agreement.

    Args:
        objects: The objects, before their volumes are carved.
        object_points: Each object's points, from its masks, by id.
        scene_points: (N, 3) leveled scene points.
        refit: Whether boxes are refit; False only scores them.
        room_yaw: Orientation of the walls, to set boxes square to them; None
            keeps the boxes' orientations.
        walls: The room's walls, to keep boxes inside them; None does not.

    Returns:
        The objects (boxes refit where it helps), and the agreement of every
        object but doors and windows, by id.
    """
    scene = scene_points[scene_points[:, 2] > FLOOR]
    result, scores = [], {}
    for obj in objects:
        own_points = object_points.get(obj.object_id)
        furniture = not (is_opening(obj) or is_wall_mounted(obj) or is_flat(obj))
        if own_points is None or len(own_points) < MIN_POINTS or is_opening(obj):
            # Too few points to score (or seen through): only kept within the walls.
            if refit and walls is not None and furniture:
                obj = replace(obj, box=_within_walls(obj.box, walls))
            result.append(obj)
            continue
        obj, scores[obj.object_id] = _agree(obj, own_points, scene,
                                            refit and furniture, room_yaw, walls)
        result.append(obj)
    return result, scores


def _agree(obj: SceneObject, own_points: FloatArray, scene: FloatArray, refit: bool,
           room_yaw: float | None, walls: Walls | None) -> tuple[SceneObject, float]:
    """Scores one object, and refits its box if asked.

    Args:
        obj: The object.
        own_points: (N, 3) its points.
        scene: (M, 3) scene points above the floor.
        refit: Whether its box may change.
        room_yaw: Orientation of the walls, or None.
        walls: The room's walls, or None.

    Returns:
        The object, and its agreement.
    """
    local, own = _neighborhood(obj.box, scene, cKDTree(own_points))
    before, _ = _f1(local, own, np.zeros(4), 0.5 * obj.box.size)
    if not refit:
        return obj, round(before, 3)
    box, score = obj.box, before
    if _refittable(obj):  # Fixed furniture keeps its box, only within the walls.
        box, score = _best_box(obj.box, before, local, own, own_points, scene, room_yaw)
    if walls is not None:
        box = _within_walls(box, walls)
    if box is not obj.box:
        local, own = _neighborhood(box, scene, cKDTree(own_points))
        score, _ = _f1(local, own, np.zeros(4), 0.5 * box.size)
        obj = replace(obj, box=box)
    return obj, round(score, 3)


def _best_box(box: OrientedBox, before: float, local: FloatArray, own: BoolArray,
              own_points: FloatArray, scene: FloatArray,
              room_yaw: float | None) -> tuple[OrientedBox, float]:
    """Chooses the box that agrees best with the camera: as it is, refit, or square to the walls.

    Args:
        box: The object's box.
        before: Its agreement.
        local: Scene points around it, in its frame.
        own: Which are the object's.
        own_points: (N, 3) the object's points.
        scene: (M, 3) scene points above the floor.
        room_yaw: Orientation of the walls; None keeps the box's.

    Returns:
        The box chosen, and its agreement.
    """
    sides, after = _climb(local, own, 0.5 * box.size)
    best = (_moved_box(box, sides), after) if after >= before + MIN_GAIN else (box, before)
    if room_yaw is None:
        return best
    square = _square_box(box, own_points, room_yaw)
    if abs(square.yaw - box.yaw) < np.radians(2.0) \
            or _core_area(own_points, square) > ALIGN_AREA * _core_area(own_points, box):
        return best
    s_local, s_own = _neighborhood(square, scene, cKDTree(own_points))
    s_sides, s_after = _climb(s_local, s_own, 0.5 * square.size)
    if s_after >= max(after, before) - ALIGN_SLACK:
        return _moved_box(square, s_sides), s_after
    return best
