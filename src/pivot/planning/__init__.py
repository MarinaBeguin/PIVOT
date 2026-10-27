"""Geometric planning: placement checks and placement search."""

from pivot.planning.placement import (
    Conflict,
    ConflictKind,
    PlacementReport,
    check_placement,
    propose_placements,
)

__all__ = [
    "Conflict",
    "ConflictKind",
    "PlacementReport",
    "check_placement",
    "propose_placements",
]
