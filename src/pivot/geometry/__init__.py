"""Geometric primitives: gravity-aligned boxes, collisions, planes and floor maps."""

from pivot.geometry.box import OrientedBox, fit_box
from pivot.geometry.collision import box_distance, box_penetration, boxes_collide
from pivot.geometry.floor import FloorMap
from pivot.geometry.plane import Plane, fit_plane_ransac, floor_alignment

__all__ = [
    "FloorMap",
    "OrientedBox",
    "Plane",
    "box_distance",
    "box_penetration",
    "boxes_collide",
    "fit_box",
    "fit_plane_ransac",
    "floor_alignment",
]
