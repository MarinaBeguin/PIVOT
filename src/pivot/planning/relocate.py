"""Moves between rooms: where an object can go in another room, and the doors on its way.

A free spot in the target room is found among the poses the object can take
there (its configuration space), preferring spots against a wall, as one
places furniture, then close ones. The path planner does the rest: it goes
through doorways like through any narrow gap, and the route tells which
rooms and doorways the object passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pivot.geometry.box import OrientedBox
from pivot.geometry.types import FloatArray
from pivot.planning.cspace import ConfigurationSpace
from pivot.planning.motion import MotionPlan, densify
from pivot.planning.placement import check_placement
from pivot.scene.graph import Scene
from pivot.scene.rooms import RoomMap

# Spacing of the points sampled along each straight piece of a path.
_SAMPLE_STEP = 0.05
# Weight of the travel distance against the gap to the wall when ranking spots.
_TRAVEL_WEIGHT = 0.15
# Poses this close to a doorway's center count as going through it.
_DOORWAY_REACH = 0.6
# Spots are kept this far from the edge of doorways, so that they stay passable.
_DOORWAY_KEEP_CLEAR = 0.6
# Exact placement checks made before giving up: when the best-ranked poses
# all fail, the thousands of others would too, at a second or so each.
_MAX_CHECKS = 60


@dataclass(frozen=True)
class Doorway:
    """A doorway a path goes through.

    Attributes:
        from_room: Name of the room it leaves.
        to_room: Name of the room it enters.
        width: Width of the passage, in meters.
        door_id: The door object there, if one was segmented.
        margin: Smallest gap between the object and the doorway's sides as
            it goes through, in meters.
    """

    from_room: str
    to_room: str
    width: float
    door_id: int | None
    margin: float

    def to_dict(self) -> dict[str, object]:
        """Serializes the doorway for tool-call responses.

        Returns:
            A JSON-compatible dictionary.
        """
        return {"from": self.from_room, "to": self.to_room, "width_m": round(self.width, 2),
                "door_id": self.door_id, "margin_m": round(self.margin, 3)}


def _rooms_along(rooms: RoomMap, plan: MotionPlan) -> list[int]:
    """Lists the rooms a motion's center passes through, in order.

    Args:
        rooms: The rooms of the scene.
        plan: The motion.

    Returns:
        Room ids, without repeats in a row.
    """
    visited: list[int] = []
    for start, end in zip(plan.waypoints, plan.waypoints[1:] or plan.waypoints):
        length = float(np.linalg.norm(end.center[:2] - start.center[:2]))
        for t in np.linspace(0.0, 1.0, max(2, math.ceil(length / _SAMPLE_STEP) + 1)):
            room = rooms.room_at(start.center[:2] + t * (end.center[:2] - start.center[:2]))
            if room and (not visited or visited[-1] != room):
                visited.append(room)
    return visited


def route(rooms: RoomMap, plan: MotionPlan) -> tuple[list[str], list[Doorway]]:
    """Lists the rooms a motion passes through, and the doorways between them.

    Args:
        rooms: The rooms of the scene.
        plan: The motion.

    Returns:
        The names of the rooms, in order, and the doorways crossed.
    """
    visited = _rooms_along(rooms, plan)
    passages = {p.rooms: p for p in rooms.passages}
    poses = densify(plan.waypoints)
    doorways = []
    for a, b in zip(visited, visited[1:]):
        passage = passages.get((min(a, b), max(a, b)))
        if passage is None:
            continue
        near = [pose for pose in poses
                if np.linalg.norm(pose.center[:2] - passage.center) <= _DOORWAY_REACH]
        margin = min((rooms.floor.distance_to_walls(pose.footprint(), search_radius=0.5)
                      for pose in near), default=math.inf)
        doorways.append(Doorway(rooms.name_of(a) or "", rooms.name_of(b) or "",
                                passage.width, passage.door_id, margin))
    return [rooms.name_of(r) or "" for r in visited], doorways


def _ranked_centers(space: ConfigurationSpace, rooms: RoomMap, room_id: int,
                    box: OrientedBox) -> list[tuple[float, FloatArray, float]]:
    """Ranks the free poses of an object in a room: against a wall and close first.

    Args:
        space: Free configurations of the object.
        rooms: The rooms of the scene.
        room_id: The target room.
        box: The object's current box.

    Returns:
        ``(score, center_xy, yaw)``, lowest score first.
    """
    floor = rooms.floor
    to_wall = ndimage.distance_transform_edt(np.pad(floor.floor, 1))[1:-1, 1:-1] * floor.resolution
    half_depth = 0.5 * float(min(box.size[:2]))
    ranked: list[tuple[float, FloatArray, float]] = []
    for k in range(space.yaw_bins):
        centers = space.free_centers(k)
        cells = np.floor((centers - floor.origin) / floor.resolution).astype(int)
        inside = ((cells[:, 0] >= 0) & (cells[:, 0] < floor.shape[1])
                  & (cells[:, 1] >= 0) & (cells[:, 1] < floor.shape[0]))
        centers, cells = centers[inside], cells[inside]
        in_room = rooms.labels[cells[:, 1], cells[:, 0]] == room_id
        centers, cells = centers[in_room], cells[in_room]
        gap = np.maximum(0.0, to_wall[cells[:, 1], cells[:, 0]] - half_depth)
        travel = np.linalg.norm(centers - box.center[:2], axis=1)
        ranked += [(float(score), center, space.yaw(k))
                   for score, center in zip(gap + _TRAVEL_WEIGHT * travel, centers)]
    ranked.sort(key=lambda item: item[0])
    return ranked


def spots_in_room(scene: Scene, object_id: int, rooms: RoomMap, room_id: int,
                  clearance: float = 0.0, max_spots: int = 8) -> list[OrientedBox]:
    """Finds free poses for an object inside a room, against a wall and close first.

    Args:
        scene: The scene.
        object_id: The object.
        rooms: The rooms of the scene.
        room_id: The target room.
        clearance: Required free distance to obstacles and walls, in meters.
        max_spots: Most poses returned.

    Returns:
        Feasible poses (exact placement checks, a bounded number of them),
        best first, out of doorways and not in front of a door; spots far
        apart from one another, so that one blocked area does not fill the list.
    """
    box = scene[object_id].box
    space = ConfigurationSpace.build(scene, object_id, clearance=clearance)
    spots: list[OrientedBox] = []
    checks = 0
    for _, center, yaw in _ranked_centers(space, rooms, room_id, box):
        if len(spots) >= max_spots or checks >= _MAX_CHECKS:
            break
        if any(np.linalg.norm(center - s.center[:2]) < 0.5 for s in spots):
            continue
        if any(np.linalg.norm(center - p.center) < 0.5 * p.width + _DOORWAY_KEEP_CLEAR
               for p in rooms.passages):
            continue
        spot = box.moved(np.asarray(center, dtype=np.float64), yaw)
        report = check_placement(scene, object_id, spot, clearance)
        checks += 1
        # A spot in front of a door would block it: not where one puts furniture.
        if report.feasible and not any("door" in w for w in report.warnings):
            spots.append(spot)
    return spots
