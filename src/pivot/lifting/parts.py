"""Joining the parts of one object that SAM tracked as separate instances.

SAM sometimes splits an object into instances that are visible together, a
chair's seat and its back for example. Masks alone cannot tell these parts
from two chairs side by side, but geometry can: parts touch in 3D, and
together they still have the size of a single object of their label.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial import cKDTree

from pivot.geometry.box import OrientedBox
from pivot.geometry.types import FloatArray

# Largest horizontal extent of one object of a label, in meters. Parts are
# only joined for these labels; anything else could be several objects.
MAX_EXTENT = {"chair": 0.85, "office chair": 0.85, "stool": 0.6, "monitor": 1.0, "lamp": 0.8,
              "trash can": 0.7, "bin": 0.7, "table": 2.0, "desk": 2.0, "shelf": 2.5,
              "cabinet": 2.0, "sofa": 2.6, "armchair": 1.1}


def join_parts(clouds: list[FloatArray], label: str, fit: Callable[[FloatArray], OrientedBox],
               max_gap: float) -> list[list[int]]:
    """Groups clouds that are touching parts of one object.

    Args:
        clouds: Point clouds of the objects of one label.
        label: Their label.
        fit: Box fitting function.
        max_gap: Largest distance between touching parts, in meters.

    Returns:
        Groups of cloud indices, each sorted, ordered by their first index.
    """
    groups = [[i] for i in range(len(clouds))]
    extent = MAX_EXTENT.get(label.lower())
    while extent is not None and (pair := _joinable(clouds, groups, fit, max_gap, extent)):
        a, b = pair
        groups[a] = sorted(groups[a] + groups.pop(b))
    return sorted(groups)


def _joinable(clouds: list[FloatArray], groups: list[list[int]],
              fit: Callable[[FloatArray], OrientedBox], max_gap: float,
              extent: float) -> tuple[int, int] | None:
    """Finds two groups that touch and together fit within one object's extent.

    Args:
        clouds: Point clouds.
        groups: Current groups of cloud indices; empty clouds never join.
        fit: Box fitting function.
        max_gap: Largest distance between touching parts, in meters.
        extent: Largest horizontal extent of one object, in meters.

    Returns:
        The positions of the two groups in ``groups``, or None.
    """
    merged = [np.vstack([clouds[i] for i in group]) for group in groups]
    for a, cloud_a in enumerate(merged):
        for b in range(a + 1, len(merged)):
            cloud_b = merged[b]
            if cloud_a.size == 0 or cloud_b.size == 0:
                continue
            gap = float(cKDTree(cloud_a).query(cloud_b)[0].min())
            if gap <= max_gap and fit(np.vstack([cloud_a, cloud_b])).size[:2].max() <= extent:
                return a, b
    return None
