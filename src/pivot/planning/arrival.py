"""A new piece of furniture arriving: it goes exactly where the user puts it.

Furniture bought new often comes flat-packed and is assembled where it will
stand: its first placement is not a move from the door, so no path is
checked, only that the spot itself is free (on the floor, clear of walls and
of other objects). If movable objects stand there, they can be moved aside,
each with a verified motion. Once there, the new object is like any other:
its later moves are planned and checked. Furniture measured in the scan never
arrives this way. Whether it would pass the doors assembled is told apart,
from the doors' widths: then it can also be delivered whole.

A new object never goes under another one (under a table, a desk): hidden
there, the user would not see it arrive. Its spot is checked against the
other objects' whole boxes, not the room left under their tops.

An object that hangs on a wall (a painting, a mirror) goes on the wall
nearest to where the user clicks, flat against it at eye level, clear of the
furniture and of the doors and windows. It moves the same way later: taken
off the wall, carried over the furniture, hung again.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from pivot.geometry.box import OrientedBox, wrap_angle
from pivot.geometry.collision import polygon_penetration
from pivot.geometry.floor import FloorMap
from pivot.geometry.types import FloatArray
from pivot.planning.motion import MotionOptions, MotionPlan
from pivot.planning.placement import ConflictKind, check_placement
from pivot.planning.rearrange import clear_spot
from pivot.scene.graph import Scene, SceneObject, is_opening, is_wall_mounted

# Margin left on each side when an object is carried through a door, in meters.
DOOR_MARGIN = 0.02
_REASONS = {
    ConflictKind.OFF_FLOOR: "it is not on the floor there",
    ConflictKind.WALL_CLEARANCE: "it would go into a wall",
    ConflictKind.CEILING: "it is taller than the ceiling there",
}
# A click this close to a wall hangs a wall object (a painting) on it, in meters.
HANG_REACH = 1.2
# Height of the center of a hung object: eye level, in meters.
HANG_HEIGHT = 1.5
# Openings this far from a hung object, across the wall, are covered by it, in meters.
_OPENING_REACH = 0.4
# Gap between a hung object and its wall, and its largest thickness, in meters.
_HANG_GAP = 0.01
_HANG_THICKNESS = 0.1


@dataclass
class Arrival:
    """Result of :func:`arrive`.

    Attributes:
        placed: Whether the object stands at the spot.
        in_the_way: Ids of the objects standing in the spot.
        steps: Verified motions of the objects moved aside, in order.
        reason: Why it was not placed, if it was not.
        box: Where it goes, when that is not where it was asked (hung on a wall).
    """

    placed: bool
    in_the_way: list[int] = field(default_factory=list)
    steps: list[MotionPlan] = field(default_factory=list)
    reason: str = ""
    box: OrientedBox | None = None


def with_new_object(scene: Scene, label: str, size: tuple[float, float, float], x: float,
                    y: float, yaw_deg: float = 0.0) -> tuple[Scene, int, OrientedBox]:
    """Adds a new object to a copy of a scene, standing on the floor at a spot.

    Args:
        scene: The scene.
        label: What it is.
        size: Width, depth and height, in meters.
        x: Where its center goes.
        y: Where its center goes.
        yaw_deg: Its orientation, in degrees.

    Returns:
        The scene with it, its id and its box.
    """
    object_id = max(scene.objects, default=0) + 1
    box = OrientedBox(np.array([x, y, 0.5 * size[2]]), np.array(size, dtype=float),
                      math.radians(yaw_deg))
    trial = Scene.from_objects([*scene.objects.values(), SceneObject(object_id, label, box)],
                               scene.floor, scene.metadata, scene.ceiling)
    return trial, object_id, box


def boxed(scene: Scene, object_id: int) -> Scene:
    """Fills the other objects' boxes, so that nothing fits under a table top.

    Args:
        scene: The scene, holding the new object.
        object_id: The new object.

    Returns:
        The scene, the other objects without their carved shapes.
    """
    return dataclasses.replace(scene, objects={
        i: o if i == object_id or o.shape is None else dataclasses.replace(o, shape=None)
        for i, o in scene.objects.items()})


def arrive(scene: Scene, object_id: int, box: OrientedBox, make_room: bool = False,
           options: MotionOptions | None = None) -> Arrival:
    """Puts a new object at an exact spot, without a path, moving others aside if asked.

    Args:
        scene: The scene, holding the new object (anywhere).
        object_id: The new object.
        box: The spot, as its box.
        make_room: Whether movable objects standing there are moved aside.
        options: Motion parameters of those moves.

    Returns:
        The arrival: placed, or what is in the way and why.
    """
    if is_wall_mounted(scene[object_id]):
        return hang(scene, object_id, float(box.center[0]), float(box.center[1]))
    report = check_placement(boxed(scene, object_id), object_id, box)
    if report.feasible:
        return Arrival(True)
    in_the_way = sorted({c.other_id for c in report.conflicts if c.other_id is not None})
    refusal = _refusal(scene, report.conflicts, in_the_way, make_room)
    if refusal:
        return Arrival(False, in_the_way, reason=refusal)
    cleared = clear_spot(scene.without([object_id]), box, in_the_way, options)
    if not cleared.success:
        return Arrival(False, in_the_way, reason=cleared.message)
    after = scene
    for step in cleared.steps:
        after = after.moved(step.object_id, step.waypoints[-1])
    if not check_placement(boxed(after, object_id), object_id, box).feasible:
        return Arrival(False, in_the_way, reason="Even with the objects moved aside, the spot is "
                                                 "not free.")
    return Arrival(True, in_the_way, cleared.steps)


def hung_box(scene: Scene, size: tuple[float, float, float], x: float,
             y: float) -> OrientedBox | None:
    """Puts a wall object on the wall nearest to a point, flat against it, at eye level.

    Args:
        scene: The scene; its floor map tells where the walls are.
        size: The object's width (along the wall), thickness and height, in meters.
        x: The point, on the floor.
        y: The point, on the floor.

    Returns:
        Its box, or None if no wall is within :data:`HANG_REACH` of a point on the floor.
    """
    floor = scene.floor
    origin = np.array([x, y], dtype=float)
    if floor is None or not floor.is_floor(origin):
        return None
    # Holes in the floor map are where furniture hid the floor, not walls.
    walled = FloorMap(floor.origin, floor.resolution, ndimage.binary_fill_holes(floor.floor))
    normal = _toward_wall(scene, walled, origin)
    if normal is None:
        return None
    # The wall's face: where the points of the wall start along that direction, or
    # where the floor ends if the walls were not seen.
    steps = np.arange(0.0, HANG_REACH + floor.resolution, 0.25 * floor.resolution)
    off = next((t for t in steps if not walled.is_floor(origin + t * normal)
                or _on_wall(floor, origin + t * normal)), None)
    if off is None:
        return None
    width, thickness = max(size[0], size[1]), min(size[0], size[1], _HANG_THICKNESS)
    center = origin + (off - 0.5 * thickness - _HANG_GAP) * normal
    height = 0.5 * size[2] + 0.05
    middle = max(HANG_HEIGHT, height)
    if scene.ceiling is not None:
        middle = min(middle, max(height, scene.ceiling - height))
    return OrientedBox(np.array([*center, middle]), np.array([width, thickness, size[2]]),
                       wrap_angle(math.atan2(normal[1], normal[0]) + 0.5 * math.pi))


def _on_wall(floor: FloorMap, point: FloatArray) -> bool:
    """Tests whether a point falls in a cell where a wall stands.

    Args:
        floor: The floor map, with its walls if known.
        point: (2,) world xy.

    Returns:
        True on a wall cell; False if the walls are unknown.
    """
    if floor.walls is None:
        return False
    col, row = np.floor((point - floor.origin) / floor.resolution).astype(int)
    rows, cols = floor.shape
    return bool(0 <= row < rows and 0 <= col < cols and floor.walls[row, col])


def room_axis(scene: Scene) -> float | None:
    """Estimates the orientation of a room's walls, modulo a quarter turn.

    Doors and windows are set in the walls when the room is built, so their
    orientation is the walls'; without them, the furniture's, which mostly
    stands parallel to the walls.

    Args:
        scene: The scene.

    Returns:
        The angle in radians, in [0, pi / 2); None for an empty scene.
    """
    openings = [o.box.yaw for o in scene.objects.values() if is_opening(o)]
    yaws = openings or [o.box.yaw for o in scene.objects.values()]
    if not yaws:
        return None
    # The mean of 4 x yaw: angles a quarter turn apart count as the same.
    mean = math.atan2(sum(math.sin(4 * a) for a in yaws), sum(math.cos(4 * a) for a in yaws))
    return (mean / 4.0) % (0.5 * math.pi)


def _toward_wall(scene: Scene, floor: FloorMap, origin: FloatArray) -> FloatArray | None:
    """Gives the direction from a point to the nearest wall, square to the room's walls.

    Rooms are taken as having square corners; a slanted wall gets the axis nearest to it.

    Args:
        scene: The scene, for the walls' orientation.
        floor: Its floor map.
        origin: (2,) the point.

    Returns:
        The unit direction; None if no wall is within :data:`HANG_REACH`.
    """
    padded = np.pad(floor.floor, 1, constant_values=False)
    distance, nearest = ndimage.distance_transform_edt(padded, return_indices=True)
    col, row = np.floor((origin - floor.origin) / floor.resolution).astype(int) + 1
    if distance[row, col] * floor.resolution > HANG_REACH:
        return None
    wall = floor.origin + (np.array([nearest[1][row, col], nearest[0][row, col]]) - 0.5) \
        * floor.resolution
    seen = math.atan2(*(wall - origin)[::-1])
    axis = room_axis(scene)
    if axis is not None:
        # Walls are taken square to the room's: the nearest cell off the floor is only
        # a few centimeters, or a few degrees, from the foot of the perpendicular.
        seen = min((axis + k * 0.5 * math.pi for k in range(4)),
                   key=lambda a: abs(math.remainder(a - seen, 2 * math.pi)))
    return np.array([math.cos(seen), math.sin(seen)])


def hang(scene: Scene, object_id: int, x: float, y: float) -> Arrival:
    """Hangs a wall object on the wall nearest to a point, if the spot there is free.

    Args:
        scene: The scene, holding the object (anywhere).
        object_id: The object.
        x: The point, on the floor.
        y: The point, on the floor.

    Returns:
        The arrival, with the object's box on the wall; or why it cannot hang there.
    """
    obj = scene[object_id]
    size = (float(obj.box.size[0]), float(obj.box.size[1]), float(obj.box.size[2]))
    box = hung_box(scene, size, x, y)
    if box is None:
        return Arrival(False, reason=f"A {obj.label} hangs on a wall: point closer to one.")
    # A door or a window of that wall: the painting would cover it. The wall seen at
    # wall height can stand a little in front of where its openings were set.
    reach = OrientedBox(box.center, np.array([box.size[0], 2.0 * _OPENING_REACH, box.size[2]]),
                        box.yaw)
    covered = [o for o in scene.objects.values() if is_opening(o) and o.box.z_max > box.z_min
               and o.box.z_min < box.z_max
               and polygon_penetration(o.box.footprint(), reach.footprint()) > 0.0]
    if covered:
        return Arrival(False, [o.object_id for o in covered],
                       reason=f"It would cover the {covered[0].label} "
                              f"(#{covered[0].object_id}).", box=box)
    report = check_placement(scene, object_id, box)
    in_the_way = sorted({c.other_id for c in report.conflicts
                         if c.kind is ConflictKind.COLLISION and c.other_id is not None})
    if in_the_way:
        names = ", ".join(f"the {scene[i].label} (#{i})" for i in in_the_way)
        verb = "stands" if len(in_the_way) == 1 else "stand"
        return Arrival(False, in_the_way, reason=f"{names} {verb} there, in front of the wall.",
                       box=box)
    if any(c.kind is ConflictKind.CEILING for c in report.conflicts):
        return Arrival(False, reason="It is taller than the wall.", box=box)
    return Arrival(True, box=box)


def _refusal(scene: Scene, conflicts: list[Any], in_the_way: list[int],
             make_room: bool) -> str:
    """Says why a spot cannot take a new object before anything is moved.

    Args:
        scene: The scene.
        conflicts: The spot's placement conflicts.
        in_the_way: Objects standing there.
        make_room: Whether movable objects may be moved aside.

    Returns:
        The reason, or an empty string if objects can be moved aside.
    """
    for conflict in conflicts:
        if conflict.kind in _REASONS:
            return f"The spot does not fit: {_REASONS[conflict.kind]}."
    fixed = [i for i in in_the_way if not scene[i].movable]
    if fixed:
        names = ", ".join(f"the {scene[i].label} (#{i})" for i in fixed)
        return f"{names} cannot be moved and stands there."
    if not make_room:
        names = ", ".join(f"the {scene[i].label} (#{i})" for i in in_the_way)
        verb = "stands" if len(in_the_way) == 1 else "stand"
        return f"{names} {verb} there: move it or make room."
    return ""


def doors_passed(scene: Scene, size: tuple[float, float, float]) -> list[dict[str, Any]]:
    """Says which doors the object would pass assembled, from their widths.

    A door's width is its long horizontal side; the object passes its
    narrowest horizontal side first, if it is not taller than the door. This
    compares sizes only: the way to the spot is not planned.

    Args:
        scene: The scene.
        size: Width, depth and height of the object, in meters.

    Returns:
        One entry per door: its id, width and whether the object fits through.
    """
    narrow = min(size[0], size[1])
    result = []
    for door in sorted(scene.objects.values(), key=lambda o: o.object_id):
        if door.label.lower() != "door":
            continue
        width = float(max(door.box.size[0], door.box.size[1]))
        result.append({"id": door.object_id, "width_m": round(width, 2),
                       "fits_assembled": narrow + 2 * DOOR_MARGIN <= width
                       and size[2] <= float(door.box.size[2])})
    return result
