"""Snapping a requested pose to the nearest feasible one.

A target picked by a click or named in words ("next to the window") is
approximate: a point a few centimeters too close to a wall should not make
the request fail. The nearest pose where the object fits is found among the
free configurations of the lattice, then confirmed with exact geometry.
"""

from __future__ import annotations

import math

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.types import FloatArray
from pivot.planning.cspace import ConfigurationSpace
from pivot.planning.placement import check_placement
from pivot.scene.graph import Scene

# A quarter turn is worth this many meters of shift when ranking candidates.
_TURN_COST = 0.2


def ranked_free_poses(space: ConfigurationSpace, center_xy: FloatArray, yaw: float,
                      allow_rotation: bool = True,
                      max_shift: float = math.inf) -> list[tuple[FloatArray, float]]:
    """Lists the free lattice poses around a pose, closest first.

    Closeness is the shift of the center plus a cost per quarter turn.

    Args:
        space: Free configurations of the object.
        center_xy: (2,) reference position, in meters.
        yaw: Reference yaw, in radians.
        allow_rotation: Whether poses may turn; otherwise they keep ``yaw``.
        max_shift: Largest shift of the center, in meters.

    Returns:
        ``(center_xy, yaw)`` pairs.
    """
    bins = range(space.yaw_bins) if allow_rotation else [space.nearest_bin(yaw)]
    ranked: list[tuple[float, FloatArray, float]] = []
    for k in bins:
        centers = space.free_centers(k)
        shift = np.linalg.norm(centers - center_xy, axis=1)
        pose_yaw = space.yaw(k) if allow_rotation else yaw
        turn = abs(math.remainder(pose_yaw - yaw, math.pi)) / (0.5 * math.pi)
        ranked += [(shift[i] + _TURN_COST * turn, centers[i], pose_yaw)
                   for i in np.flatnonzero(shift <= max_shift)]
    ranked.sort(key=lambda item: item[0])
    return [(center, pose_yaw) for _, center, pose_yaw in ranked]


def nearest_feasible_pose(scene: Scene, object_id: int, goal: OrientedBox, max_shift: float,
                          clearance: float = 0.0, allow_rotation: bool = True,
                          max_checks: int = 60) -> OrientedBox | None:
    """Finds the feasible pose closest to a requested one.

    Args:
        scene: The scene.
        object_id: The object to place.
        goal: The requested pose.
        max_shift: Largest distance the object's center may move from the
            requested position, in meters.
        clearance: Required free distance to obstacles and walls, in meters.
        allow_rotation: Whether the yaw may change; otherwise it is kept.
        max_checks: Maximum number of exact placement checks.

    Returns:
        The requested pose if it is feasible, else the nearest feasible pose
        within ``max_shift``, or None if there is none.
    """
    if check_placement(scene, object_id, goal, clearance).feasible:
        return goal
    space = ConfigurationSpace.build(scene, object_id, clearance=clearance)
    candidates = ranked_free_poses(space, goal.center[:2], goal.yaw, allow_rotation, max_shift)
    for center, yaw in candidates[:max_checks]:
        box = goal.moved(center, yaw)
        if check_placement(scene, object_id, box, clearance).feasible:
            return box
    return None
