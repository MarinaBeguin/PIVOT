"""Placement verification and search: the agent's geometric oracle.

The language model proposes, this module disposes. ``check_placement`` explains
precisely why a pose is infeasible (which object it hits and by how much, how
much of it hangs off the floor), and ``propose_placements`` searches for
feasible poses next to a reference object.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox, rotation_2d, wrap_angle
from pivot.geometry.collision import (
    box_distance_lower_bound, polygon_distance, polygon_penetration,
)
from pivot.geometry.solid import solid_distance, solid_penetration
from pivot.geometry.floor import FloorMap
from pivot.geometry.types import FloatArray
from pivot.scene.graph import Scene, SceneObject, carried_boxes, carried_objects, is_opening


class ConflictKind(str, Enum):
    """Reasons a placement can fail."""

    COLLISION = "collision"
    CLEARANCE = "clearance"
    WALL_CLEARANCE = "wall_clearance"
    OFF_FLOOR = "off_floor"
    CEILING = "ceiling"
    NOT_MOVABLE = "not_movable"


@dataclass(frozen=True)
class Conflict:
    """One reason a placement is infeasible.

    Attributes:
        kind: Category of the conflict.
        amount: Magnitude in meters (penetration depth for ``COLLISION``,
            remaining gap for the clearance kinds, height above the ceiling
            for ``CEILING``) or, for ``OFF_FLOOR``, the fraction of the
            footprint off the floor.
        other_id: The object involved, if any.
        other_label: Its label, for human-readable explanations.
        carried_id: When the conflict is caused by an object carried by the
            moved one (a monitor on a desk), that object's id.
        carried_label: Its label.
    """

    kind: ConflictKind
    amount: float
    other_id: int | None = None
    other_label: str | None = None
    carried_id: int | None = None
    carried_label: str | None = None

    def describe(self) -> str:
        """Renders the conflict as a short English sentence.

        Returns:
            A human-readable description.
        """
        other = f"the {self.other_label} (#{self.other_id})" if self.other_id is not None else ""
        subject = (f"the {self.carried_label} (#{self.carried_id}) it carries "
                   if self.carried_id is not None else "")
        if self.kind is ConflictKind.COLLISION:
            return f"{subject}overlaps {other} by {100 * self.amount:.0f} cm"
        if self.kind is ConflictKind.CLEARANCE:
            return f"{subject}leaves only {100 * self.amount:.0f} cm of clearance to {other}"
        if self.kind is ConflictKind.WALL_CLEARANCE:
            return f"leaves only {100 * self.amount:.0f} cm of clearance to a wall"
        if self.kind is ConflictKind.OFF_FLOOR:
            return f"{100 * self.amount:.0f}% of its footprint is outside the walkable floor"
        if self.kind is ConflictKind.CEILING:
            return f"{subject}would rise {100 * self.amount:.0f} cm above the ceiling"
        return "is a fixed part of the room and cannot be moved"

    def to_dict(self) -> dict[str, Any]:
        """Serializes the conflict with machine-readable magnitudes.

        Returns:
            A JSON-compatible dictionary.
        """
        result: dict[str, Any] = {"kind": self.kind.value, "description": self.describe()}
        if self.kind is ConflictKind.COLLISION:
            result["penetration_m"] = round(self.amount, 3)
        elif self.kind in (ConflictKind.CLEARANCE, ConflictKind.WALL_CLEARANCE):
            result["gap_m"] = round(self.amount, 3)
        elif self.kind is ConflictKind.OFF_FLOOR:
            result["off_floor_fraction"] = round(self.amount, 3)
        elif self.kind is ConflictKind.CEILING:
            result["above_ceiling_m"] = round(self.amount, 3)
        if self.other_id is not None:
            result["other_id"] = self.other_id
            result["other_label"] = self.other_label
        if self.carried_id is not None:
            result["carried_id"] = self.carried_id
            result["carried_label"] = self.carried_label
        return result


@dataclass(frozen=True)
class PlacementReport:
    """Outcome of checking one candidate pose.

    Attributes:
        object_id: The object being placed.
        box: Its box at the candidate pose.
        conflicts: Every detected problem; empty if feasible.
        gap_to_anchor: Footprint distance to the reference object, if any.
        travel: Distance from the object's current position, in meters.
        warnings: Remarks that do not make the pose infeasible (the object
            stands in front of a window).
    """

    object_id: int
    box: OrientedBox
    conflicts: list[Conflict] = field(default_factory=list)
    gap_to_anchor: float | None = None
    travel: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        """Whether the pose has no conflict."""
        return not self.conflicts

    def to_dict(self) -> dict[str, Any]:
        """Serializes the report for tool-call responses.

        Returns:
            A compact JSON-compatible dictionary.
        """
        result: dict[str, Any] = {
            "object_id": self.object_id,
            "feasible": self.feasible,
            "center_xy": [round(float(v), 3) for v in self.box.center[:2]],
            "yaw_deg": round(math.degrees(self.box.yaw), 1),
            "travel_m": round(self.travel, 3),
            "conflicts": [c.to_dict() for c in self.conflicts],
        }
        if self.gap_to_anchor is not None:
            result["gap_to_anchor_m"] = round(self.gap_to_anchor, 3)
        if self.warnings:
            result["warnings"] = self.warnings
        return result


def shrink_footprint(box: OrientedBox, margin: float) -> FloatArray:
    """Returns a box's footprint shrunk by a margin on every side.

    Args:
        box: The box.
        margin: Shrinking distance, in meters.

    Returns:
        The (4, 2) shrunk footprint, counter-clockwise.
    """
    size = box.size.copy()
    size[:2] = np.maximum(size[:2] - 2.0 * margin, 0.0)
    return OrientedBox(box.center, size, box.yaw).footprint()


def _floor_conflicts(floor: FloorMap, box: OrientedBox, clearance: float,
                     floor_margin: float) -> list[Conflict]:
    """Checks that a footprint stands on the floor, away from walls.

    Args:
        floor: The floor map.
        box: The candidate box.
        clearance: Required distance to walls, in meters.
        floor_margin: Margin of the on-floor test, see ``check_placement``.

    Returns:
        At most one conflict: off the floor, or too close to a wall.
    """
    fraction = floor.floor_fraction(shrink_footprint(box, floor_margin))
    if fraction < 1.0 - 1e-9:
        return [Conflict(ConflictKind.OFF_FLOOR, 1.0 - fraction)]
    # The footprint grown by the clearance must stay on the floor too.
    if clearance > 0.0 and floor.floor_fraction(
            shrink_footprint(box, floor_margin - clearance)) < 1.0 - 1e-9:
        gap = floor.distance_to_walls(box.footprint(), search_radius=clearance)
        return [Conflict(ConflictKind.WALL_CLEARANCE, 0.0 if math.isinf(gap) else gap)]
    return []


def check_placement(
    scene: Scene,
    object_id: int,
    box: OrientedBox,
    clearance: float = 0.0,
    floor_margin: float = 0.025,
    collision_tolerance: float = 0.01,
    carried: Collection[int] | None = None,
) -> PlacementReport:
    """Checks whether an object can be placed at a given pose.

    Objects resting on the moved object travel with it and are ignored.
    Collisions are tested on the objects' voxel shapes when they have one, so
    a carton can stand under a table top; otherwise on their boxes, which also
    lets a chair slide under a wall-mounted shelf.

    Args:
        scene: Current scene.
        object_id: The object to place.
        box: Its proposed box. Must keep the object's size.
        clearance: Required free distance to every other object and to walls,
            in meters.
        floor_margin: The footprint, shrunk by this distance on every side,
            must lie entirely on walkable floor. Absorbs floor map
            discretization without letting objects cross thin walls.
        collision_tolerance: Interpenetration depth tolerated, absorbing the
            noise of boxes fitted to reconstructed points.
        carried: Ids of the objects resting on the moved object, if already
            known. Computed from the scene otherwise.

    Returns:
        A report listing every conflict.
    """
    obj = scene[object_id]
    conflicts: list[Conflict] = []
    if not obj.movable:
        conflicts.append(Conflict(ConflictKind.NOT_MOVABLE, 0.0))

    if carried is None:
        carried = carried_objects(scene, object_id)
    ignored = {object_id, *carried}
    others = [o for o in scene.objects.values()
              if o.object_id not in ignored and not is_opening(o)]
    conflicts += _object_conflicts(others, obj, box, clearance, collision_tolerance)
    child_boxes = carried_boxes(scene, object_id, box) if carried else {}
    for child_id in carried:
        conflicts += _object_conflicts(others, scene[child_id], child_boxes[child_id],
                                       clearance, collision_tolerance, carried=True)
    if scene.ceiling is not None:
        conflicts += _ceiling_conflicts(scene, box, child_boxes,
                                        scene.ceiling + collision_tolerance)

    if scene.floor is not None:
        conflicts.extend(_floor_conflicts(scene.floor, box, clearance, floor_margin))

    travel = float(np.linalg.norm(box.center[:2] - obj.box.center[:2]))
    return PlacementReport(object_id, box, conflicts, travel=travel,
                           warnings=opening_warnings(scene, box))


def _ceiling_conflicts(scene: Scene, box: OrientedBox, child_boxes: dict[int, OrientedBox],
                       ceiling: float) -> list[Conflict]:
    """Checks that an object, and what it carries, stays under the ceiling.

    Args:
        scene: The scene.
        box: The object's proposed box.
        child_boxes: Boxes of the objects it carries, keyed by id.
        ceiling: Highest height allowed, in meters.

    Returns:
        One conflict per box rising above the ceiling.
    """
    conflicts = []
    if box.z_max > ceiling:
        conflicts.append(Conflict(ConflictKind.CEILING, box.z_max - ceiling))
    for child_id, child in child_boxes.items():
        if child.z_max > ceiling:
            conflicts.append(Conflict(ConflictKind.CEILING, child.z_max - ceiling,
                                      carried_id=child_id, carried_label=scene[child_id].label))
    return conflicts


# How far in front of an opening an object is in the way: the passage of a
# door, the reach to a window.
OPENING_REACH = {"door": 0.8, "window": 0.3}


def front_zone(opening: OrientedBox, reach: float) -> OrientedBox:
    """Returns the zone in front of and behind an opening, as wide as the opening.

    The opening's long horizontal side is its width; the zone extends along
    the other side, on both faces (which one opens onto the room is unknown).

    Args:
        opening: Box of the door or window.
        reach: Depth of the zone on each face, in meters.

    Returns:
        The zone, on the floor, as tall as the opening.
    """
    size = opening.size.copy()
    depth_axis = 0 if size[0] < size[1] else 1
    size[depth_axis] += 2.0 * reach
    return OrientedBox(opening.center.copy(), size, opening.yaw)


def opening_warnings(scene: Scene, box: OrientedBox) -> list[str]:
    """Notes the doors and windows an object would stand in front of.

    Standing beside an opening, against the same wall, is not in front of it.

    Args:
        scene: The scene.
        box: The object's box.

    Returns:
        One sentence per opening.
    """
    footprint = box.footprint()
    return [f"stands in front of the {o.label} (#{o.object_id})"
            for o in scene.objects.values()
            if is_opening(o) and polygon_penetration(footprint, front_zone(
                o.box, OPENING_REACH.get(o.label.lower(), 0.3)).footprint()) > 0.01]


def _object_conflicts(others: list[SceneObject], mover: SceneObject, box: OrientedBox,
                      clearance: float, tolerance: float, carried: bool = False
                      ) -> list[Conflict]:
    """Lists the collisions and clearance violations of one moved object.

    Args:
        others: Objects that do not move.
        mover: The moved object (or an object it carries).
        box: Its box at the tested pose.
        clearance: Required free distance, in meters.
        tolerance: Interpenetration depth tolerated, in meters.
        carried: Whether ``mover`` is carried by the object being placed.

    Returns:
        The conflicts.
    """
    carried_id, carried_label = (mover.object_id, mover.label) if carried else (None, None)
    conflicts = []
    for other in others:
        # Objects certainly farther than the clearance can neither collide nor be too close.
        if box_distance_lower_bound(box, other.box) > clearance:
            continue
        depth = solid_penetration(box, mover.shape, other.box, other.shape, tolerance)
        if depth > tolerance:
            conflicts.append(Conflict(ConflictKind.COLLISION, depth, other.object_id,
                                      other.label, carried_id, carried_label))
        elif clearance > 0.0:
            distance = solid_distance(box, mover.shape, other.box, other.shape, horizon=clearance)
            if distance < clearance:
                conflicts.append(Conflict(ConflictKind.CLEARANCE, distance, other.object_id,
                                          other.label, carried_id, carried_label))
    return conflicts


def _side_positions(half_anchor: FloatArray, half_mover: FloatArray, gap: float,
                    samples_per_side: int) -> FloatArray:
    """Places a mover beside the four sides of an anchor, in the anchor frame.

    Args:
        half_anchor: (2,) half extents of the anchor footprint.
        half_mover: (2,) half extents of the mover footprint along the anchor axes.
        gap: Distance between the facing sides.
        samples_per_side: Number of evenly spaced offsets along each side.

    Returns:
        (4 * samples_per_side, 2) mover centers relative to the anchor center.
    """
    positions = []
    for axis in (0, 1):
        other = 1 - axis
        slide = half_anchor[other] + half_mover[other]
        for sign in (-1.0, 1.0):
            side = np.zeros((samples_per_side, 2))
            side[:, axis] = sign * (half_anchor[axis] + half_mover[axis] + gap)
            side[:, other] = np.linspace(-slide, slide, samples_per_side)
            positions.append(side)
    result: FloatArray = np.vstack(positions)
    return result


def candidate_poses(
    mover: OrientedBox, anchor: OrientedBox, gaps: tuple[float, ...], yaws: tuple[float, ...],
    samples_per_side: int = 5,
) -> list[OrientedBox]:
    """Generates poses that put a box beside each side of an anchor box.

    For each side of the anchor, each yaw and each gap, the mover is slid
    along the side at evenly spaced offsets, at a distance chosen so that the
    footprints are exactly ``gap`` apart when axis-aligned with the anchor.

    Args:
        mover: The box to place, at its current pose.
        anchor: The reference box.
        gaps: Target footprint distances to the anchor, in meters.
        yaws: Absolute yaws to try for the mover, in radians.
        samples_per_side: Number of offsets along each side.

    Returns:
        Candidate boxes, keeping the mover's height.
    """
    rotation = rotation_2d(anchor.yaw)
    half_anchor = 0.5 * anchor.size[:2]
    candidates: list[OrientedBox] = []
    for yaw in yaws:
        # Half extents of the mover's footprint along the anchor's axes.
        half_mover = 0.5 * np.abs(rotation_2d(yaw - anchor.yaw)) @ mover.size[:2]
        for gap in gaps:
            local = _side_positions(half_anchor, half_mover, gap, samples_per_side)
            for center in local @ rotation.T + anchor.center[:2]:
                candidates.append(mover.moved(center, yaw))
    return candidates


def _spread_out(reports: list[PlacementReport], spacing: float,
                max_results: int) -> list[PlacementReport]:
    """Greedily keeps the best reports whose centers are mutually far apart.

    Args:
        reports: Reports sorted best first.
        spacing: Minimum distance between two kept centers, in meters.
        max_results: Maximum number of reports to keep.

    Returns:
        The kept reports, in their original order.
    """
    selected: list[PlacementReport] = []
    for report in reports:
        if len(selected) == max_results:
            break
        if all(np.linalg.norm(report.box.center[:2] - s.box.center[:2]) >= spacing
               for s in selected):
            selected.append(report)
    return selected


def propose_placements(
    scene: Scene,
    object_id: int,
    anchor_id: int,
    max_gap: float = 0.4,
    clearance: float = 0.0,
    max_results: int = 3,
    floor_margin: float = 0.025,
) -> list[PlacementReport]:
    """Searches for feasible poses that put an object next to another.

    Candidates are ranked by footprint distance to the anchor first, then by
    how far the object has to travel. Results are spread out: two proposals are
    never closer than the object's smaller horizontal side.

    Args:
        scene: Current scene.
        object_id: The object to place.
        anchor_id: The object to place it next to.
        max_gap: Largest footprint distance to the anchor that still counts as
            "next to", in meters.
        clearance: Required free distance to every other object, in meters.
        max_results: Maximum number of proposals.
        floor_margin: The footprint, shrunk by this distance on every side,
            must lie entirely on walkable floor. Absorbs floor map
            discretization without letting objects cross thin walls.

    Returns:
        Up to ``max_results`` feasible reports, best first. Empty if none exists.
    """
    mover, anchor = scene[object_id].box, scene[anchor_id].box
    # Keep the current yaw, or align with the anchor in any of its four directions.
    yaws = tuple(dict.fromkeys(
        round(wrap_angle(y), 6)
        for y in (mover.yaw, *(anchor.yaw + 0.5 * math.pi * k for k in range(4)))))
    gaps = tuple(float(g) for g in np.linspace(0.05, max_gap, 4))
    feasible: list[PlacementReport] = []
    for box in candidate_poses(mover, anchor, gaps, yaws):
        report = check_placement(scene, object_id, box, clearance, floor_margin)
        gap = polygon_distance(box.footprint(), anchor.footprint())
        if report.feasible and gap <= max_gap:
            feasible.append(PlacementReport(object_id, box, [], gap, report.travel))

    feasible.sort(key=lambda r: (round(r.gap_to_anchor or 0.0, 2), r.travel))
    return _spread_out(feasible, float(min(mover.size[0], mover.size[1])), max_results)
