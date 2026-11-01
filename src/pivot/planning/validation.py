"""Exact checks of an object's motions, the ground truth of the motion planner.

A pose is checked with :func:`pivot.planning.placement.check_placement`; a
straight motion (a translation combined with a steady turn) by checking poses
along it, at most 2 cm and 2 degrees apart. Poses that a cheap bound proves
free are skipped, and results are remembered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from pivot.geometry.box import OrientedBox, wrap_angle
from pivot.geometry.collision import interval_gap, rectangle_gaps
from pivot.geometry.types import FloatArray
from pivot.planning.placement import ConflictKind, PlacementReport, check_placement
from pivot.scene.graph import Scene, carried_objects, obstructs


def yaw_step(yaw_a: float, yaw_b: float) -> float:
    """Returns the smallest rotation between two rectangle orientations.

    Rectangles are symmetric under a half turn, so the result lies in
    [-pi/2, pi/2].

    Args:
        yaw_a: Start yaw, in radians.
        yaw_b: End yaw, in radians.

    Returns:
        The signed rotation, in radians.
    """
    delta = wrap_angle(yaw_b - yaw_a)
    if delta > 0.5 * math.pi:
        delta -= math.pi
    elif delta < -0.5 * math.pi:
        delta += math.pi
    return delta


@dataclass(frozen=True)
class Validator:
    """Exact collision checks of straight motions between poses.

    Attributes:
        scene: The scene.
        object_id: The moving object.
        clearance: Required distance to other objects, in meters.
        floor_margin: Margin of the on-floor test, see ``check_placement``.
        max_step: Maximum translation between two checked poses, in meters.
        max_turn: Maximum rotation between two checked poses, in radians.
        carried: Ids of the objects resting on the moving object.
    """

    scene: Scene
    object_id: int
    clearance: float
    floor_margin: float
    max_step: float = 0.02
    max_turn: float = math.radians(2.0)
    carried: tuple[int, ...] = ()

    @classmethod
    def create(cls, scene: Scene, object_id: int, clearance: float,
               floor_margin: float) -> Validator:
        """Builds a validator, computing the carried objects once.

        Args:
            scene: The scene.
            object_id: The moving object.
            clearance: Required distance to other objects, in meters.
            floor_margin: Margin of the on-floor test, see ``check_placement``.

        Returns:
            The validator.
        """
        return cls(scene, object_id, clearance, floor_margin,
                   carried=tuple(carried_objects(scene, object_id)))

    def pose_ok(self, box: OrientedBox) -> bool:
        """Checks one pose exactly.

        Args:
            box: The object's box at that pose.

        Returns:
            True if the pose has no conflict.
        """
        return (self.free_distance(box) > 0.0
                or check_placement(self.scene, self.object_id, box, self.clearance,
                                   self.floor_margin, carried=self.carried).feasible)

    @cached_property
    def _obstacles(self) -> tuple[FloatArray, FloatArray, FloatArray] | None:
        """Obstacles of the moving object, for :meth:`free_distance`.

        Returns:
            The (N, 4, 2) footprints of the objects that can come within the
            clearance of the moving object, their (N, 2) centers and the (N,)
            radii of their circumscribed circles; None when the bound does not
            apply (the object carries others, is fixed, or rises above the
            ceiling).
        """
        mover = self.scene[self.object_id]
        ceiling = self.scene.ceiling
        if self.carried or not mover.movable or (ceiling is not None
                                                 and mover.box.z_max > ceiling):
            return None
        box = mover.box
        # Sliding never changes heights: objects above or below stay out of reach.
        boxes = [o.box for o in self.scene.objects.values()
                 if o.object_id != self.object_id and obstructs(o, mover)
                 and interval_gap(box.z_min, box.z_max, o.box.z_min, o.box.z_max)
                 <= self.clearance]
        return (np.reshape([b.footprint() for b in boxes], (-1, 4, 2)),
                np.reshape([b.center[:2] for b in boxes], (-1, 2)),
                np.array([0.5 * math.hypot(*b.size[:2]) for b in boxes]))

    def free_distance(self, box: OrientedBox) -> float:
        """Bounds from below how far the object can move from a pose and stay free.

        Uses the boxes of the obstacles, which contain their shapes, and the
        floor map. If the result is positive, the pose is free under
        :meth:`pose_ok`, and so is every pose reached by moving no point of
        the footprint farther than the result.

        Args:
            box: The object's box at the pose.

        Returns:
            The distance in meters; 0 or less when the exact test is needed.
        """
        obstacles = self._obstacles
        if obstacles is None or abs(box.z_min - self.scene[self.object_id].box.z_min) > 1e-6:
            return 0.0  # The obstacles were selected at the object's own height.
        footprints, centers, radii = obstacles
        # Circles bound the gaps from below; only the close ones are worth measuring.
        circles = (np.hypot(*(centers - box.center[:2]).T) - radii
                   - 0.5 * math.hypot(*box.size[:2]))
        near = circles < _EXACT_HORIZON
        free = min(float(np.min(circles[~near], initial=math.inf)),
                   float(np.min(rectangle_gaps(box.footprint(), footprints[near]),
                                initial=math.inf)))
        free -= self.clearance
        floor = self.scene.floor
        if floor is not None:
            # The exact test lets the footprint, shrunk by the floor margin and
            # grown by the clearance, reach the floor's edge; this one does not.
            free = min(free, floor.open_distance(box)
                       - max(0.0, self.clearance - self.floor_margin))
        return free

    @cached_property
    def _checked(self) -> dict[tuple[float, ...], bool]:
        """Results of :meth:`segment_ok`, keyed by the motion's endpoints."""
        return {}

    def segment_ok(self, start: OrientedBox, end: OrientedBox) -> bool:
        """Checks a straight translation combined with a rotation, exactly.

        Results are remembered: repaired searches revisit the same steps.

        Args:
            start: Box at the start of the motion.
            end: Box at its end.

        Returns:
            True if every interpolated pose is free.
        """
        key = (*np.round(start.center, 6), round(start.yaw, 6),
               *np.round(end.center, 6), round(end.yaw, 6))
        known = self._checked.get(key)
        if known is None:
            known = self._checked[key] = self._segment_ok(start, end)
        return known

    def _segment_ok(self, start: OrientedBox, end: OrientedBox) -> bool:
        """Checks a straight motion, without memory; see :meth:`segment_ok`.

        Args:
            start: Box at the start of the motion.
            end: Box at its end.

        Returns:
            True if every interpolated pose is free.
        """
        turn = yaw_step(start.yaw, end.yaw)
        distance = float(np.linalg.norm(end.center[:2] - start.center[:2]))
        steps = max(1, math.ceil(distance / self.max_step), math.ceil(abs(turn) / self.max_turn))
        poses = [start.moved((1 - t) * start.center[:2] + t * end.center[:2], start.yaw + t * turn)
                 for t in np.linspace(0.0, 1.0, steps + 1)[1:]]
        index = 0
        while index < len(poses):
            free = self.free_distance(poses[index])
            if free > 0.0:
                # No point of the footprint moves that far until a later pose: skip.
                reached = poses[index]
                index += 1
                while index < len(poses) and displacement(reached, poses[index]) < free:
                    index += 1
                continue
            report = self._report(poses[index])
            if not report.feasible:
                # Voxel aliasing: two volumes sliding past each other can overlap
                # by one cell at a single intermediate pose. A real contact lasts.
                last = index == len(poses) - 1
                if last or not aliasing(report) or not self.pose_ok(poses[index + 1]):
                    return False
                index += 1
            index += 1
        return True

    def _report(self, box: OrientedBox) -> PlacementReport:
        """Checks one pose exactly, with the details of its conflicts.

        Args:
            box: The object's box at that pose.

        Returns:
            The placement report.
        """
        return check_placement(self.scene, self.object_id, box, self.clearance,
                               self.floor_margin, carried=self.carried)


# Obstacles whose circle is this close to the object's have their exact gap measured.
_EXACT_HORIZON = 0.5


def displacement(a: OrientedBox, b: OrientedBox) -> float:
    """Bounds how far any point of a footprint moves between two poses.

    Args:
        a: First pose.
        b: Second pose.

    Returns:
        The translation plus the rotation times the half diagonal, in meters.
    """
    return (float(np.linalg.norm(b.center[:2] - a.center[:2]))
            + 0.5 * math.hypot(*a.size[:2]) * abs(yaw_step(a.yaw, b.yaw)))


# Volumes are 4 cm cells: an isolated overlap up to this depth is a sampling artifact.
ALIASING_DEPTH = 0.05


def aliasing(report: PlacementReport) -> bool:
    """Tells whether a pose's conflicts could be voxel aliasing only.

    Args:
        report: The pose's placement report.

    Returns:
        True if every conflict is a collision at most one cell deep.
    """
    return all(c.kind is ConflictKind.COLLISION and c.amount <= ALIASING_DEPTH
               for c in report.conflicts)
