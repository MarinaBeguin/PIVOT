"""Clearing a way for a person or a wheelchair to a place, moving as few objects as possible.

"Make room for a pupil in a wheelchair at table 24" asks for two things: a
free spot at the table, as big as the wheelchair, and a passage as wide as
the wheelchair from a door to that spot. In a classroom, many sets of
chairs could be moved for it; the one that moves the fewest is wanted.

The passage is searched on the floor grid as the path of a disk as wide as
the passage, from the doors to each candidate spot along the sides of the
target. Walls and fixed objects stop it; a movable object only makes a cell
costly, so the cheapest path avoids crossing furniture when a detour of
reasonable length exists. Among the cheapest spots, the one whose passage
and spot overlap the fewest movable objects is kept. Those objects are then
parked out of the way with verified motions, and the passage is checked
again with every object in place.
"""

from __future__ import annotations

import math
from functools import partial
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import polygon_penetration
from pivot.geometry.floor import FloorMap
from pivot.geometry.types import BoolArray, FloatArray, IntArray
from pivot.planning.handling import can_be_carried
from pivot.planning.access import check_access, obstacle_mask, preserves_doors
from pivot.planning.bottlenecks import route_gaps
from pivot.planning.motion import MotionOptions, MotionPlan
from pivot.planning.placement import OPENING_REACH, front_zone
from pivot.planning.rearrange import park_aside, restore_group
from pivot.scene.graph import Scene, SceneObject, is_flat, is_opening, support_parents

WHEELCHAIR_SPOT = (1.0, 0.8)
"""Depth (away from the table) and width of the spot a wheelchair takes at a table, in
meters: a wheelchair is about 1.1 m long, the knees go under the table top."""

# Objects lower than this stop a person or a wheelchair; higher ones hang on a wall.
_REACH_HEIGHT = 1.0
# Extra cost of a cell covered by a movable object, per meter of path: crossing a
# chair costs as much as a detour about this many times as long.
_MOVE_PENALTY = 10.0
# Spots along each side of the target, this far apart, in meters.
_SPOT_STEP = 0.2
# Gap between the spot and the target, in meters.
_SPOT_GAP = 0.02
# Cheapest spots whose blockers are counted exactly.
_CANDIDATES = 8
# Moving an object that cannot be carried (a table) counts as this many light ones.
_HEAVY = 3
# Largest number of objects moved aside.
_MAX_MOVED = 8
# Spots whose objects in the way are tried, fewest objects first.
_ATTEMPTS = 3
# Length of the boxes the passage is cut into, in meters.
_PIECE = 0.25


@dataclass
class Way:
    """Result of :func:`plan_way`.

    Attributes:
        success: Whether the way is clear after the steps.
        message: What was done, or why it failed.
        spot: The spot at the target, if one was found.
        path: (N, 2) points of the passage, from a door to the spot.
        moved: Objects parked out of the way, in order.
        steps: Their verified motions; empty on failure.
        door_id: The door the passage starts from.
        restored: Group members returned towards their original pose while retaining access.
    """

    success: bool
    message: str
    spot: OrientedBox | None = None
    path: FloatArray = field(default_factory=lambda: np.zeros((0, 2)))
    moved: list[int] = field(default_factory=list)
    steps: list[MotionPlan] = field(default_factory=list)
    door_id: int | None = None
    restored: list[int] = field(default_factory=list)

    def to_dict(self, scene: Scene, width: float) -> dict[str, Any]:
        """Serializes the result for tool-call responses.

        Args:
            scene: The scene before the moves (for labels).
            width: Width of the passage, in meters.

        Returns:
            A JSON-compatible dictionary.
        """
        result: dict[str, Any] = {"success": self.success, "message": self.message,
                                  "returned_toward_original": self.restored,
                                  "validation_scope": "Reconstruction and grid; disk passage "
                                  "and reserved spot, not complete wheelchair ergonomics.",
                                  "passage_width_m": width,
                                  "moved_aside": [{"id": i, "label": scene[i].label}
                                                  for i in self.moved]}
        if self.spot is not None:
            result["spot"] = {"center_xy": [round(float(v), 3) for v in self.spot.center[:2]],
                              "yaw_deg": round(math.degrees(self.spot.yaw), 1),
                              "size_m": [round(float(v), 2) for v in self.spot.size[:2]]}
        if len(self.path):
            path = self.path
            if self.door_id is not None:
                path = np.vstack([scene[self.door_id].box.center[:2], path])
            if self.spot is not None:
                path = np.vstack([path, self.spot.center[:2]])
            result["path_xy"] = path.tolist()
            result["path_length_m"] = round(float(np.linalg.norm(
                np.diff(path, axis=0), axis=1).sum()), 2)
            result["from_door"] = self.door_id
            if not self.success and self.spot is not None:
                # Describe the attempted route, not a fabricated universal cause.
                result["candidate_bottlenecks"] = route_gaps(scene, self.path, width, -1)
        return result


def _obstacles(scene: Scene, target_id: int) -> tuple[list[SceneObject], list[SceneObject]]:
    """Splits what stops a person into fixed objects and objects that can be moved aside.

    Objects resting on another one move with it; the target stays.

    Args:
        scene: The scene.
        target_id: The object the way leads to.

    Returns:
        ``(fixed, movable)``.
    """
    parents = support_parents(scene)
    fixed, movable = [], []
    for obj in scene.objects.values():
        if is_opening(obj) or is_flat(obj) or obj.box.z_min >= _REACH_HEIGHT \
                or obj.object_id in parents:
            continue
        if obj.movable and obj.object_id != target_id:
            movable.append(obj)
        else:
            fixed.append(obj)
    return fixed, movable


def _inflated(floor: FloorMap, obj: SceneObject, radius: float) -> BoolArray:
    """Rasterizes an object's footprint grown by a radius.

    Args:
        floor: The grid.
        obj: The object.
        radius: Growth, in meters.

    Returns:
        A (rows, cols) mask.
    """
    if obj.shape is None:
        size = obj.box.size.copy()
        size[:2] += 2.0 * radius
        return floor.rasterize(OrientedBox(obj.box.center, size, obj.box.yaw).footprint())
    occupied = obstacle_mask(floor, obj, _REACH_HEIGHT)
    if not occupied.any():
        return occupied
    grown: BoolArray = ndimage.maximum_filter(
        occupied, size=2 * int(np.ceil(radius / floor.resolution)) + 1)
    return grown


def _walkable(floor: FloorMap, radius: float) -> BoolArray:
    """Finds the cells where a disk of a radius stands on the floor, away from walls.

    Args:
        floor: The floor map.
        radius: The disk's radius, in meters.

    Returns:
        A (rows, cols) mask.
    """
    padded = np.pad(floor.floor, 1, constant_values=False)
    distance = ndimage.distance_transform_edt(padded)[1:-1, 1:-1] * floor.resolution
    result: BoolArray = distance >= radius
    return result


def _edges(free: BoolArray, weight: FloatArray) -> sparse.csr_matrix:
    """Links each free cell to its free neighbors, the cost being that of entering one.

    Args:
        free: (rows, cols) cells a path may use.
        weight: (rows, cols) cost of entering each cell, per cell step.

    Returns:
        The (cells, cells) graph.
    """
    rows, cols = free.shape
    index = np.arange(rows * cols).reshape(rows, cols)
    links: list[tuple[IntArray, IntArray, FloatArray]] = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        # Each cell and its neighbor at (dr, dc), linked both ways.
        left, right = max(0, -dc), cols - max(0, dc)
        a = index[:rows - dr, left:right].ravel()
        b = index[dr:, left + dc:right + dc].ravel()
        both = free.ravel()[a] & free.ravel()[b]
        a, b = a[both], b[both]
        links += [(a, b, math.hypot(dr, dc) * weight.ravel()[b]),
                  (b, a, math.hypot(dr, dc) * weight.ravel()[a])]
    return sparse.csr_matrix((np.concatenate([cost for _, _, cost in links]),
                              (np.concatenate([u for u, _, _ in links]),
                               np.concatenate([v for _, v, _ in links]))),
                             shape=(rows * cols, rows * cols))


def _distances(free: BoolArray, weight: FloatArray, sources: IntArray
               ) -> tuple[FloatArray, IntArray]:
    """Computes the cheapest paths from source cells over the free cells.

    Args:
        free: (rows, cols) cells a path may use.
        weight: (rows, cols) cost of entering each cell, per cell step.
        sources: Flat indices of the start cells.

    Returns:
        The cost of reaching each cell (inf where unreachable) and each
        cell's predecessor (negative for none), both flat.
    """
    cost, previous, _ = csgraph.dijkstra(_edges(free, weight), indices=sources, min_only=True,
                                         return_predecessors=True)
    return cost, previous


def _spots(target: OrientedBox, size: tuple[float, float], gap: float = _SPOT_GAP
           ) -> list[OrientedBox]:
    """Lists the spots along the sides of an object, facing it.

    Args:
        target: The object's box.
        size: Depth and width of a spot, in meters.
        gap: Separation from the table, including representation uncertainty.

    Returns:
        Boxes on the floor, their depth (x axis) pointing away from the object.
    """
    depth, width = size
    corners = target.footprint()
    spots = []
    for start, end in zip(corners, np.roll(corners, -1, axis=0)):
        length = float(np.linalg.norm(end - start))
        along = (end - start) / max(length, 1e-9)
        normal = np.array([along[1], -along[0]])
        if np.dot(normal, 0.5 * (start + end) - target.center[:2]) < 0:
            normal = -normal
        offsets = np.arange(0.5 * width, length - 0.5 * width + 1e-9, _SPOT_STEP) \
            if length > width else np.array([0.5 * length])
        for offset in offsets:
            center = start + offset * along + (gap + 0.5 * depth) * normal
            spots.append(OrientedBox(np.array([*center, 0.5 * _REACH_HEIGHT]),
                                     np.array([depth, width, _REACH_HEIGHT]),
                                     math.atan2(normal[1], normal[0])))
    return spots


def _cell(floor: FloorMap, point: FloatArray) -> tuple[int, int]:
    """Returns the (row, col) of the cell holding a point.

    Args:
        floor: The grid.
        point: (2,) world xy.

    Returns:
        The cell indices, possibly out of bounds.
    """
    cell = np.floor((point - floor.origin) / floor.resolution).astype(int)
    return int(cell[1]), int(cell[0])


def _trace(floor: FloorMap, previous: IntArray, end: int) -> FloatArray:
    """Follows predecessors back from a cell to a source.

    Args:
        floor: The grid.
        previous: Flat predecessor of each cell.
        end: Flat index of the last cell.

    Returns:
        (N, 2) cell centers, from the source to the end.
    """
    cells = [end]
    while previous[cells[-1]] >= 0:
        cells.append(int(previous[cells[-1]]))
    rows_cols = np.unravel_index(np.array(cells[::-1]), floor.shape)
    result: FloatArray = floor.origin + (np.column_stack(rows_cols[::-1]) + 0.5) \
        * floor.resolution
    return result


def _corridor(path: FloatArray, spot: OrientedBox, width: float) -> list[OrientedBox]:
    """Cuts a passage and its spot into boxes, for overlap tests.

    Args:
        path: (N, 2) points of the passage.
        spot: The spot at its end.
        width: Width of the passage, in meters.

    Returns:
        Boxes as high as a person reaches.
    """
    keep = [path[0]]
    for point in path[1:]:
        if np.linalg.norm(point - keep[-1]) >= _PIECE:
            keep.append(point)
    keep.append(path[-1])
    boxes = [spot]
    for start, end in zip(keep, keep[1:]):
        length = float(np.linalg.norm(end - start))
        if length < 1e-6:
            continue
        middle = 0.5 * (start + end)
        boxes.append(OrientedBox(np.array([*middle, 0.5 * _REACH_HEIGHT]),
                                 np.array([length + width, width, _REACH_HEIGHT]),
                                 math.atan2(*(end - start)[::-1])))
    return boxes


def _in_the_way(movable: list[SceneObject], corridor: list[OrientedBox],
                 floor: FloorMap | None = None) -> list[int]:
    """Lists the movable objects a passage overlaps, in the order the passage meets them.

    Args:
        movable: The movable objects.
        corridor: The passage's boxes, the spot first.
        floor: Optional grid for conservative occupied-shape intersections.

    Returns:
        Their ids.
    """
    found: list[int] = []
    masks = ({obj.object_id: obstacle_mask(floor, obj, _REACH_HEIGHT)
              for obj in movable if obj.shape is not None} if floor is not None else {})
    for piece in [*corridor[1:], corridor[0]]:
        occupied = floor.rasterize(piece.footprint()) if floor is not None and masks else None
        for obj in movable:
            if obj.object_id not in found and polygon_penetration(
                    obj.box.footprint(), piece.footprint()) > 0.01:
                if occupied is not None and obj.object_id in masks \
                        and not (masks[obj.object_id] & occupied).any():
                    continue
                found.append(obj.object_id)
    return found


@dataclass(frozen=True)
class _Search:
    """The cheapest passages from the doors, and what can be in the way."""

    cost: FloatArray
    previous: IntArray
    free: BoolArray
    movable: list[SceneObject]
    fixed_area: BoolArray
    doors: dict[int, int]


def _search(scene: Scene, target_id: int, radius: float) -> _Search | None:
    """Computes the cheapest passages from the doors for a disk.

    Args:
        scene: The scene; must have a floor map.
        target_id: The object the way leads to.
        radius: Half the passage width, in meters.

    Returns:
        The search, or None if no door has free floor in front of it.
    """
    floor = scene.floor
    assert floor is not None
    fixed, movable = _obstacles(scene, target_id)
    blocked = np.zeros(floor.shape, dtype=bool)
    for obj in fixed:
        blocked |= _inflated(floor, obj, radius)
    free = _walkable(floor, radius) & ~blocked
    covered = np.zeros(floor.shape, dtype=np.float64)
    for obj in movable:
        covered += _inflated(floor, obj, radius)
    fixed_area = np.zeros(floor.shape, dtype=bool)
    for obj in fixed:
        fixed_area |= obstacle_mask(floor, obj, _REACH_HEIGHT)
    doors = _door_cells(scene, free, 2 * radius)
    blocked_doors = {door for door in set(doors.values())
                     if (floor.rasterize(front_zone(scene[door].box,
                                                    OPENING_REACH["door"]).footprint())
                         & fixed_area).any()}
    doors = {cell: door for cell, door in doors.items() if door not in blocked_doors}
    if not doors:
        return None
    cost, previous = _distances(free, floor.resolution * (1.0 + _MOVE_PENALTY * covered),
                                np.array(sorted(doors)))
    return _Search(cost, previous, free, movable, fixed_area, doors)


def _door_cells(scene: Scene, free: BoolArray, width: float = 0.0) -> dict[int, int]:
    """Lists the free cells in front of each door, where a passage can start.

    Args:
        scene: The scene; must have a floor map.
        free: (rows, cols) cells a passage may use.
        width: Minimum reconstructed doorway width, in meters.

    Returns:
        The door of each such cell, by flat cell index.
    """
    floor = scene.floor
    assert floor is not None
    doors: dict[int, int] = {}
    for door in (o for o in scene.objects.values() if o.label.lower() == "door"):
        if max(door.box.size[:2]) + 1e-9 < width:
            continue
        cells = np.flatnonzero(floor.rasterize(front_zone(door.box, 0.8).footprint()) & free)
        doors.update({int(c): door.object_id for c in cells})
    return doors


def _entry(floor: FloorMap, search: _Search, spot: OrientedBox) -> int | None:
    """Returns the cell a passage reaches a spot by, if the spot is usable and reached.

    Args:
        floor: The floor map.
        search: The passages from the doors.
        spot: The spot.

    Returns:
        The flat index of its center cell, or None.
    """
    row, col = _cell(floor, spot.center[:2])
    if not (0 <= row < floor.shape[0] and 0 <= col < floor.shape[1]) \
            or not search.free[row, col] \
            or floor.floor_fraction(spot.footprint()) < 0.99 \
            or (floor.rasterize(spot.footprint()) & search.fixed_area).any():
        return None
    flat = row * floor.shape[1] + col
    return flat if np.isfinite(search.cost[flat]) else None


def _door_corridor(scene: Scene, path: FloatArray, spot: OrientedBox, width: float,
                   door: int | None) -> list[OrientedBox]:
    """Includes the actual door approach, not just an interior route seed.

    Args:
        scene: Complete scene.
        path: Proposed route.
        spot: Destination footprint.
        width: Passage width in meters.
        door: Starting door identifier, if known.

    Returns:
        Route footprint plus the entrance zone that must also be cleared.
    """
    corridor = _corridor(path, spot, width)
    if door is not None:
        corridor.append(front_zone(scene[door].box, OPENING_REACH["door"]))
    return corridor


def _ranked_spots(scene: Scene, search: _Search, target_id: int, size: tuple[float, float],
                  width: float, facing_object_id: int | None = None
                  ) -> list[tuple[OrientedBox, FloatArray, list[int]]]:
    """Ranks the spots by the objects their passage moves, fewest and lightest first.

    Args:
        scene: The scene.
        search: The passages from the doors.
        target_id: The object the way leads to.
        size: Depth and width of the spot, in meters.
        width: Width of the passage, in meters.
        facing_object_id: Optional reference the seated person should face.

    Returns:
        Each spot with its passage and the objects in the way.
    """
    floor = scene.floor
    assert floor is not None
    candidates = []
    for spot in _spots(scene[target_id].box, size, _spot_gap(scene[target_id], floor)):
        if not _faces(scene, spot, facing_object_id):
            continue
        flat = _entry(floor, search, spot)
        if flat is not None:
            candidates.append((float(search.cost[flat]), flat, spot))
    ranked = []
    for cost, flat, spot in sorted(candidates, key=lambda c: c[0])[:_CANDIDATES]:
        path = _trace(floor, search.previous, flat)
        blockers = _in_the_way(search.movable, _door_corridor(
            scene, path, spot, width,
            search.doors.get(int(np.ravel_multi_index(_cell(floor, path[0]), floor.shape)))), floor)
        effort = sum(1 if can_be_carried(scene[b]) else _HEAVY for b in blockers)
        ranked.append(((effort, cost), spot, path, blockers))
    ranked.sort(key=lambda item: item[0])
    return [(spot, path, blockers) for _, spot, path, blockers in ranked]


def _spot_gap(target: SceneObject, floor: FloorMap) -> float:
    """Separates a place from the conservative raster margin of its own table.

    Args:
        target: Furniture at the place.
        floor: Grid defining spatial uncertainty.

    Returns:
        Gap in meters without shrinking the obstacle or the wheelchair.
    """
    if target.shape is None or not target.shape:
        return _SPOT_GAP
    return (_SPOT_GAP + target.shape.resolution * math.sqrt(3) / 2
            + (1 + math.sqrt(2)) * floor.resolution)


def clear_way(scene: Scene, target_id: int, width: float = 0.9,
              spot_size: tuple[float, float] = WHEELCHAIR_SPOT,
              facing_object_id: int | None = None) -> Way:
    """Finds an already clear route and spot without preparing furniture moves.

    A facing reference filters places to those facing within 45 degrees of
    that object. This tests plan orientation, not ergonomics or knee clearance.

    Args:
        scene: Current layout.
        target_id: Furniture to reach.
        width: Passage width in meters.
        spot_size: Depth and width of the place.
        facing_object_id: Optional facing reference.

    Returns:
        A verified existing route or a failed bounded search, without edits.
    """
    failure = Way(False, "No clear route and spot found in the current layout.")
    if scene.floor is not None and target_id in scene.objects:
        search = _search(scene, target_id, 0.5 * width)
        if search is not None:
            for spot, path, blockers in _ranked_spots(
                    scene, search, target_id, spot_size, width, facing_object_id):
                door = search.doors.get(int(np.ravel_multi_index(
                    _cell(scene.floor, path[0]), scene.floor.shape)))
                if not blockers:
                    return Way(True, "Route and spot are clear in the current layout.",
                               spot, path, door_id=door)
                if failure.spot is None:
                    failure = Way(False, "Candidate only: furniture moves have not been verified.",
                                  spot, path, blockers, door_id=door)
    return failure


def _faces(scene: Scene, spot: OrientedBox, reference: int | None) -> bool:
    """Checks whether the seated direction points towards an identified reference."""
    if reference is None:
        return True
    direction = scene[reference].box.center[:2] - spot.center[:2]
    facing = -np.array([math.cos(spot.yaw), math.sin(spot.yaw)])
    return bool(np.dot(facing, direction) >= np.linalg.norm(direction) / math.sqrt(2))


def way_is_clear(scene: Scene, target_id: int, width: float = 0.9,
                 spot_size: tuple[float, float] = WHEELCHAIR_SPOT,
                 facing_object_id: int | None = None) -> bool:
    """Tests whether a person or a wheelchair already gets to an object, nothing moved.

    Args:
        scene: The scene; must have a floor map.
        target_id: The object to reach (a table).
        width: Width of the passage, in meters.
        spot_size: Depth and width of the spot at the object, in meters.
        facing_object_id: Optional facing reference.

    Returns:
        True if some spot at the object is free and reached by a free passage.
    """
    return clear_way(scene, target_id, width, spot_size, facing_object_id).success


def plan_way(scene: Scene, target_id: int, width: float = 0.9,
             spot_size: tuple[float, float] = WHEELCHAIR_SPOT,
             options: MotionOptions | None = None,
             facing_object_id: int | None = None) -> Way:
    """Plans a passage from a door to a spot at an object, moving as few objects as possible.

    Args:
        scene: The scene; must have a floor map.
        target_id: The object the way leads to (a table).
        width: Width of the passage, in meters (0.9 for a wheelchair).
        spot_size: Depth and width of the spot at the object, in meters.
        options: Motion parameters of the objects moved aside.
        facing_object_id: Optional facing reference.

    Returns:
        The way: the spot, the passage, and the verified motions that clear them.
    """
    options = options or MotionOptions()
    if scene.floor is None:
        return Way(False, "This scene has no floor map.")
    if not any(obj.label.lower() == "door" for obj in scene.objects.values()):
        return Way(False, "No door was reconstructed; an entrance route cannot be verified.")
    search = _search(scene, target_id, 0.5 * width)
    if search is None:
        return Way(False, "No door has free floor in front of it.")
    ranked = _ranked_spots(scene, search, target_id, spot_size, width, facing_object_id)
    if not ranked:
        return Way(False, f"No verified candidate route to a {spot_size[1]:.2f} x "
                          f"{spot_size[0]:.2f} m spot at this {scene[target_id].label} was "
                          "found with the current reconstruction, grid and orientation.")
    failure = Way(False, "No spot could be cleared.")
    # Objects that could not be parked: spots needing them moved are skipped.
    stuck: set[int] = set()
    attempts = 0
    for spot, path, blockers in ranked:
        if stuck & set(blockers):
            continue
        way = _clear_way(scene, target_id, Way(False, "", spot, path, blockers,
                         door_id=search.doors.get(int(np.ravel_multi_index(
                             _cell(scene.floor, path[0]), scene.floor.shape)))),
                         width, options)
        if way.success:
            return way
        if len(way.moved) < len(blockers):
            stuck.add(blockers[len(way.moved)])
        failure = way if failure.spot is None else failure
        attempts += 1
        if attempts == _ATTEMPTS:
            break
    return failure


def _clear_way(scene: Scene, target_id: int, way: Way, width: float,
               options: MotionOptions) -> Way:
    """Parks the objects in the way of a passage, then checks it.

    Args:
        scene: The scene.
        target_id: The object the way leads to.
        way: The spot, passage and objects in the way (``moved`` lists them).
        width: Width of the passage, in meters.
        options: Motion parameters.

    Returns:
        The way with its motions on success; else why it failed, ``moved``
        holding the objects parked before the failure.
    """
    assert way.spot is not None
    if len(way.moved) > _MAX_MOVED:
        return Way(False, f"{len(way.moved)} objects are in the way; at most {_MAX_MOVED} are "
                          "moved aside.", way.spot, way.path, door_id=way.door_id)
    corridor = _door_corridor(scene, way.path, way.spot, width, way.door_id)
    steps: list[MotionPlan] = []
    # The place is planned at this table's current pose. A pushed row must not
    # silently take the target along and leave the reserved place behind.
    cleared = replace(scene, objects={**scene.objects,
                                     target_id: replace(scene[target_id], movable=False)})
    # Tables first: pushed back with their chairs, they often clear those too.
    for blocker in sorted(way.moved, key=lambda b: can_be_carried(scene[b])):
        if any(step.object_id == blocker for step in steps) \
                and not _in_the_way([cleared[blocker]], corridor, cleared.floor):
            # Already pushed out of the way with its table.
            continue
        parking = park_aside(cleared, blocker, corridor, options, prefer_edges=True,
            accept=partial(_keeps_access, scene,
                           [blocker,*(step.object_id for step in steps)], width))
        if parking is None:
            return Way(False, f"The {scene[blocker].label} (#{blocker}) is in the way and no "
                              "verified parking place was found nearby preserving the passage "
                              "and existing circulation.", way.spot, way.path,
                       [s.object_id for s in steps], door_id=way.door_id)
        for motion in parking:
            steps.append(motion)
            cleared = cleared.moved(motion.object_id, motion.waypoints[-1])
    cleared,steps,restored = _restore_heavy(scene,cleared,steps,corridor,width,options)
    still = _in_the_way(_obstacles(cleared, target_id)[1], corridor, cleared.floor)
    if still:
        return Way(False, "After moving the objects in the way, the passage is still not "
                          f"clear: {_names(scene, still)} in it.", way.spot, way.path,
                   way.moved, door_id=way.door_id)
    moved = list(dict.fromkeys(step.object_id for step in steps))
    if not preserves_doors(scene, cleared, width):
        return Way(False, "Clearing this route would obstruct another door; no moves executed.")
    lost = _lost_access(scene, cleared, moved, width)
    if lost:
        return Way(False, "This layout would lose existing access to "
                   f"{_names(scene, lost)}; no moves executed.")
    return Way(True, f"Clear passage, {width:.2f} m wide, from door #{way.door_id} to the "
                     f"spot; moved aside: {_names(scene, moved) or 'nothing'}.", way.spot,
               way.path, moved, steps, way.door_id, restored)


def _restore_heavy(initial: Scene, current: Scene, steps: list[MotionPlan],
                   corridor: list[OrientedBox], width: float, options: MotionOptions
                   ) -> tuple[Scene,list[MotionPlan],list[int]]:
    """Reduces final displacement when furniture can move back without losing access."""
    moved = list(dict.fromkeys(step.object_id for step in steps))
    restored: list[int] = []
    for object_id in reversed(moved):
        if can_be_carried(initial[object_id]):
            continue
        returned = restore_group(current,initial,object_id,corridor,options,
                                  partial(_keeps_access,initial,moved,width))
        if returned is not None:
            for step in returned:
                current = current.moved(step.object_id,step.waypoints[-1])
            steps += returned
            restored.extend(step.object_id for step in returned)
    return current,steps,list(dict.fromkeys(restored))


def _lost_access(initial: Scene, final: Scene, moved: list[int], width: float) -> list[int]:
    """Finds newly unreachable furniture at walking and requested route widths.

    Args:
        initial: Layout before rearrangement.
        final: Proposed layout.
        moved: Objects parked aside.
        width: Requested passage width.

    Returns:
        Lost accessible objects, allowing light blockers to be stored out of reach.
    """
    lost: set[int] = set()
    for access_width in (0.6, width):
        before, after = check_access(initial, access_width), check_access(final, access_width)
        assert before is not None and after is not None
        lost.update(i for i in set(after.unreachable) - set(before.unreachable)
                    if not can_be_carried(initial[i]) or i not in moved)
    return sorted(lost)


def _keeps_access(initial: Scene, moved: list[int], width: float, final: Scene) -> bool:
    """Filters candidate parking layouts that would damage existing circulation.

    Args:
        initial: Original layout.
        moved: Objects deliberately parked aside.
        width: Requested passage width.
        final: Candidate layout before motion execution.

    Returns:
        Whether door approaches and previous furniture access are preserved.
    """
    return preserves_doors(initial,final,width) and not _lost_access(initial,final,moved,width)


def _names(scene: Scene, ids: list[int]) -> str:
    """Names objects for a message.

    Args:
        scene: The scene.
        ids: Their ids.

    Returns:
        "the chair (#2), the table (#5)".
    """
    return ", ".join(f"the {scene[i].label} (#{i})" for i in ids)
