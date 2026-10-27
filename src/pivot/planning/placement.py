"""Placement verification and search: the agent's geometric oracle.

The language model proposes, this module disposes. ``check_placement`` explains
precisely why a pose is infeasible (which object it hits and by how much, how
much of it hangs off the floor), and ``propose_placements`` searches for
feasible poses next to a reference object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox, rotation_2d, wrap_angle
from pivot.geometry.collision import box_distance, box_penetration, polygon_distance
from pivot.geometry.types import FloatArray
from pivot.scene.graph import Scene, carried_objects


class ConflictKind(str, Enum):
    """Reasons a placement can fail."""

    COLLISION = "collision"
    CLEARANCE = "clearance"
    OFF_FLOOR = "off_floor"
    NOT_MOVABLE = "not_movable"


@dataclass(frozen=True)
class Conflict:
    """One reason a placement is infeasible.

    Attributes:
        kind: Category of the conflict.
        amount: Magnitude in meters (penetration depth, missing clearance) or,
            for ``OFF_FLOOR``, the fraction of the footprint off the floor.
        other_id: The object involved, if any.
        other_label: Its label, for human-readable explanations.
    """

    kind: ConflictKind
    amount: float
    other_id: int | None = None
    other_label: str | None = None

    def describe(self) -> str:
        """Renders the conflict as a short English sentence.

        Returns:
            A human-readable description.
        """
        other = f"the {self.other_label} (#{self.other_id})" if self.other_id is not None else ""
        if self.kind is ConflictKind.COLLISION:
            return f"overlaps {other} by {100 * self.amount:.0f} cm"
        if self.kind is ConflictKind.CLEARANCE:
            return f"leaves only {100 * self.amount:.0f} cm of clearance to {other}"
        if self.kind is ConflictKind.OFF_FLOOR:
            return f"{100 * self.amount:.0f}% of its footprint is outside the walkable floor"
        return "is a fixed part of the room and cannot be moved"


@dataclass(frozen=True)
class PlacementReport:
    """Outcome of checking one candidate pose.

    Attributes:
        object_id: The object being placed.
        box: Its box at the candidate pose.
        conflicts: Every detected problem; empty if feasible.
        gap_to_anchor: Footprint distance to the reference object, if any.
        travel: Distance from the object's current position, in meters.
    """

    object_id: int
    box: OrientedBox
    conflicts: list[Conflict] = field(default_factory=list)
    gap_to_anchor: float | None = None
    travel: float = 0.0

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
            "conflicts": [c.describe() for c in self.conflicts],
        }
        if self.gap_to_anchor is not None:
            result["gap_to_anchor_m"] = round(self.gap_to_anchor, 3)
        return result


def check_placement(
    scene: Scene,
    object_id: int,
    box: OrientedBox,
    clearance: float = 0.0,
    min_floor_fraction: float = 0.95,
    collision_tolerance: float = 0.01,
) -> PlacementReport:
    """Checks whether an object can be placed at a given pose.

    Objects resting on the moved object travel with it and are ignored, as are
    objects that do not intersect the moved object's height range (a chair can
    slide under a wall-mounted shelf).

    Args:
        scene: Current scene.
        object_id: The object to place.
        box: Its proposed box. Must keep the object's size.
        clearance: Required free distance to every other object, in meters.
        min_floor_fraction: Minimum fraction of the footprint on walkable floor.
        collision_tolerance: Interpenetration depth tolerated, absorbing the
            noise of boxes fitted to reconstructed points.

    Returns:
        A report listing every conflict.
    """
    obj = scene[object_id]
    conflicts: list[Conflict] = []
    if not obj.movable:
        conflicts.append(Conflict(ConflictKind.NOT_MOVABLE, 0.0))

    ignored = {object_id, *carried_objects(scene, object_id)}
    for other in scene.objects.values():
        if other.object_id in ignored:
            continue
        depth = box_penetration(box, other.box)
        if depth > collision_tolerance:
            conflicts.append(Conflict(ConflictKind.COLLISION, depth, other.object_id, other.label))
        elif clearance > 0.0:
            distance = box_distance(box, other.box)
            if distance < clearance:
                conflicts.append(
                    Conflict(ConflictKind.CLEARANCE, distance, other.object_id, other.label))

    if scene.floor is not None:
        fraction = scene.floor.floor_fraction(box.footprint())
        if fraction < min_floor_fraction:
            conflicts.append(Conflict(ConflictKind.OFF_FLOOR, 1.0 - fraction))

    travel = float(np.linalg.norm(box.center[:2] - obj.box.center[:2]))
    return PlacementReport(object_id, box, conflicts, travel=travel)


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
    min_floor_fraction: float = 0.95,
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
        min_floor_fraction: Minimum fraction of the footprint on walkable floor.

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
        report = check_placement(scene, object_id, box, clearance, min_floor_fraction)
        gap = polygon_distance(box.footprint(), anchor.footprint())
        if report.feasible and gap <= max_gap:
            feasible.append(PlacementReport(object_id, box, [], gap, report.travel))

    feasible.sort(key=lambda r: (round(r.gap_to_anchor or 0.0, 2), r.travel))
    return _spread_out(feasible, float(min(mover.size[0], mover.size[1])), max_results)
