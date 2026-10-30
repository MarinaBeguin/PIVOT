"""Accessibility: can a person (or a wheelchair) still get around the room?

A rearrangement can be collision-free and still leave no way to the window, or
a desk walled off by chairs. On the floor map, the cells a person of a given
width can stand on are those at least half that width from any wall or
object they cannot pass under. Starting from the doors, the reachable cells
are the connected ones; an object is accessible if one can stand within arm's
reach of it. An object resting on another (a monitor on a desk) is reached
through it, so only its support is judged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage

from pivot.geometry.types import BoolArray, IntArray
from pivot.planning.placement import front_zone
from pivot.scene.graph import Scene, is_opening, support_parents

# Walking width of an adult, and of a wheelchair (accessible-route guidelines).
WALKING_WIDTH = 0.6
WHEELCHAIR_WIDTH = 0.9
# One can walk under what is higher than this (a wall shelf), not under a table.
HEADROOM = 1.9
# How close one must stand to an object to use or grab it.
ARM_REACH = 0.6


@dataclass(frozen=True)
class AccessReport:
    """Which parts of a room a person of a given width can reach.

    Attributes:
        width: Width of the person, in meters.
        from_doors: Whether the walk starts at the doors (else anywhere).
        reachable: Mask of the floor map cells the person can stand on.
        reachable_share: Share of the free floor (clear of objects) a
            person walks on, as a fraction.
        blocked_doors: Ids of the doors with no room to stand in front.
        unreachable: Ids of the objects nobody can get within arm's reach of.
        labels: Labels of the objects named above, by id.
    """

    width: float
    from_doors: bool
    reachable: BoolArray
    reachable_share: float
    blocked_doors: list[int] = field(default_factory=list)
    unreachable: list[int] = field(default_factory=list)
    labels: dict[int, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the report for tool-call responses.

        Returns:
            A JSON-compatible dictionary.
        """
        return {
            "width_m": self.width,
            "from_doors": self.from_doors,
            "reachable_floor_share": round(self.reachable_share, 3),
            "blocked_doors": [{"id": i, "label": self.labels[i]} for i in self.blocked_doors],
            "unreachable_objects": [{"id": i, "label": self.labels[i]}
                                    for i in self.unreachable],
            "accessible": not self.blocked_doors and not self.unreachable,
        }


def _standable(scene: Scene, width: float) -> tuple[BoolArray, BoolArray]:
    """Finds where a person of a given width can stand.

    Args:
        scene: The scene; must have a floor map.
        width: Width the person needs, in meters.

    Returns:
        The free floor cells (clear of objects one cannot pass under), and
        those at least half the width away from any wall or such object.
    """
    floor = scene.floor
    assert floor is not None
    obstacles = np.zeros(floor.shape, dtype=bool)
    for obj in scene.objects.values():
        if not is_opening(obj) and obj.box.z_min < HEADROOM:
            obstacles |= floor.rasterize(obj.box.footprint())
    free = floor.floor & ~obstacles
    # Distance, in meters, from each free cell to the nearest wall or object;
    # beyond the map's edge is no floor either.
    clearance = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * floor.resolution
    return free, clearance >= 0.5 * width


def _start_components(scene: Scene, components: IntArray) -> tuple[set[int], list[int]]:
    """Finds the areas a walk starts from: those in front of the doors.

    Args:
        scene: The scene; must have a floor map.
        components: Labels of the connected areas one can stand in (0: none).

    Returns:
        The labels of the areas reached from a door, and the ids of the doors
        with no area in front.
    """
    floor = scene.floor
    assert floor is not None
    starts: set[int] = set()
    blocked: list[int] = []
    for door in (o for o in scene.objects.values() if o.label.lower() == "door"):
        zone = floor.rasterize(front_zone(door.box, 0.8).footprint())
        found = set(np.unique(components[zone]).tolist()) - {0}
        if not found:
            blocked.append(door.object_id)
        starts |= found
    return starts, blocked


def check_access(scene: Scene, width: float = WALKING_WIDTH) -> AccessReport | None:
    """Finds the floor a person of a given width can reach from the doors.

    Args:
        scene: The scene; needs a floor map.
        width: Width the person needs to pass, in meters.

    Returns:
        The report, or None if the scene has no floor map.
    """
    floor = scene.floor
    if floor is None:
        return None
    free, standable = _standable(scene, width)
    components, _ = ndimage.label(standable)
    starts, blocked = _start_components(scene, components)
    from_doors = bool(starts)
    if not from_doors:
        # Without a usable door, the walk starts in the largest open area.
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        starts = {int(np.argmax(sizes))} if sizes.max() > 0 else set()
    reachable = np.isin(components, list(starts)) & standable

    # Cells a person standing on reachable cells can touch.
    reach = ndimage.binary_dilation(
        reachable, iterations=max(1, round((0.5 * width + ARM_REACH) / floor.resolution)))
    resting = support_parents(scene)
    unreachable = [
        o.object_id for o in sorted(scene.objects.values(), key=lambda o: o.object_id)
        if not is_opening(o) and o.object_id not in resting
        and not (reach & floor.rasterize(o.box.footprint())).any()]
    # The floor a person walks on is wider than the cells their center visits.
    walked = ndimage.binary_dilation(
        reachable, iterations=max(1, round(0.5 * width / floor.resolution))) & free
    return AccessReport(width, from_doors, reachable,
                        float(walked.sum()) / max(int(free.sum()), 1), blocked, unreachable,
                        {o.object_id: o.label for o in scene.objects.values()})
