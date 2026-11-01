"""Moving instructions: the verified moves of a session, as steps a person can follow.

Each step names the object, how far it travels, where it ends (next to the
nearest object, as one would say it), whether it turns, whether it is carried
rather than slid, and how many people it takes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import polygon_distance
from pivot.planning.handling import people_needed
from pivot.planning.motion import MotionPlan
from pivot.planning.relocate import Doorway, route
from pivot.scene.graph import Scene
from pivot.scene.rooms import RoomMap, find_rooms

# Below these, a turn or a lift is not worth mentioning.
_MIN_TURN_DEG = 10.0
_MIN_LIFT = 0.05


@dataclass(frozen=True)
class Step:
    """One move of the instructions.

    Attributes:
        number: Position in the sequence, from 1.
        object_id: The moved object.
        label: Its label.
        start: Its box before the move.
        end: Its box after the move.
        distance: Length of the path, in meters.
        carried: Whether it is lifted rather than slid along the floor.
        people: How many people it takes.
        landmark: The nearest other object at the end, if any: its label, its
            id and the gap to it, in meters.
        doorways: The doorways it goes through, when it changes room.
    """

    number: int
    object_id: int
    label: str
    start: OrientedBox
    end: OrientedBox
    distance: float
    carried: bool
    people: int
    landmark: tuple[str, int, float] | None
    doorways: tuple[Doorway, ...] = ()

    @property
    def turn_deg(self) -> float:
        """Signed change of orientation, counterclockwise, in degrees."""
        return math.degrees(math.remainder(self.end.yaw - self.start.yaw, 2 * math.pi))

    def text(self) -> str:
        """Renders the step as one English sentence.

        Returns:
            The instruction.
        """
        verb = "Carry" if self.carried else "Slide"
        parts = [f"{verb} the {self.label} (#{self.object_id}) {_length(self.distance)}"]
        if self.doorways:
            first, last = self.doorways[0], self.doorways[-1]
            doors = ", ".join(_doorway(d) for d in self.doorways)
            parts[0] += f" from the {first.from_room} to the {last.to_room} ({doors})"
        if self.landmark is not None:
            label, landmark_id, gap = self.landmark
            where = "against" if gap < 0.03 else f"{_length(gap)} from"
            parts.append(f"to {where} the {label} (#{landmark_id})")
        if abs(self.turn_deg) >= _MIN_TURN_DEG:
            side = "counterclockwise" if self.turn_deg > 0 else "clockwise"
            parts.append(f"turning it {abs(self.turn_deg):.0f} degrees {side}")
        sentence = ", ".join([" ".join(parts[:2]), *parts[2:]]) + "."
        if self.people > 1:
            sentence += " Two people needed."
        return sentence


def _doorway(doorway: Doorway) -> str:
    """Says which doorway is crossed and with how much room to spare.

    Args:
        doorway: The doorway.

    Returns:
        E.g. ``"door #6, 5 cm to spare"``.
    """
    name = f"door #{doorway.door_id}" if doorway.door_id is not None else "doorway"
    if math.isfinite(doorway.margin):
        return f"{name}, {100 * doorway.margin:.0f} cm to spare"
    return name


def _length(meters: float) -> str:
    """Formats a length as one would say it.

    Args:
        meters: The length.

    Returns:
        Centimeters below a meter, else meters with one decimal.
    """
    return f"{100 * meters:.0f} cm" if meters < 1.0 else f"{meters:.1f} m"


def _nearest(scene: Scene, object_id: int, box: OrientedBox,
             ignored: set[int]) -> tuple[str, int, float] | None:
    """Finds the object nearest to a footprint, to say where something ends.

    Args:
        scene: The scene after the move.
        object_id: The moved object.
        box: Its box.
        ignored: Objects that moved along with it.

    Returns:
        The landmark's label and id, and the footprint gap to it; None if
        the scene has no other object.
    """
    candidates = [(other.label, other.object_id,
                   polygon_distance(box.footprint(), other.box.footprint()))
                  for other in scene.objects.values()
                  if other.object_id != object_id and other.object_id not in ignored]
    return min(candidates, key=lambda c: c[2]) if candidates else None


def instructions(initial: Scene, motions: list[MotionPlan]) -> list[Step]:
    """Turns executed motions into moving instructions.

    Args:
        initial: The scene before the first motion.
        motions: The motions, in order of execution.

    Returns:
        One step per motion.
    """
    steps = []
    scene = initial
    rooms = find_rooms(initial)
    if rooms is not None and len(rooms.rooms) < 2:
        rooms = None
    for number, motion in enumerate(motions, start=1):
        after = scene.moved(motion.object_id, motion.waypoints[-1])
        steps.append(_step(number, scene, after, motion, rooms))
        scene = after
    return steps


def _step(number: int, before: Scene, after: Scene, motion: MotionPlan,
          rooms: RoomMap | None) -> Step:
    """Describes one executed motion.

    Args:
        number: Its position in the sequence, from 1.
        before: The scene before it.
        after: The scene after it.
        motion: The motion.
        rooms: The rooms, when the home has several.

    Returns:
        The step.
    """
    obj = before[motion.object_id]
    start, end = motion.waypoints[0], motion.waypoints[-1]
    moved_along = {i for i in before.objects if i != obj.object_id
                   and not np.allclose(after[i].box.center, before[i].box.center)}
    lowest = min(start.z_min, end.z_min)
    return Step(number, obj.object_id, obj.label, start, end, motion.length,
                carried=any(w.z_min > lowest + _MIN_LIFT for w in motion.waypoints),
                people=people_needed(obj),
                landmark=_nearest(after, obj.object_id, end, moved_along),
                doorways=tuple(route(rooms, motion)[1]) if rooms else ())
