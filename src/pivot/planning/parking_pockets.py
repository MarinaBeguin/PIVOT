"""Furniture storage pockets that leave a pedestrian strip on their open side."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from pivot.geometry.box import OrientedBox
from pivot.geometry.floor import FloorMap
from pivot.geometry.types import BoolArray, FloatArray, IntArray
from pivot.planning.access import HEADROOM, WALKING_WIDTH, obstacle_mask
from pivot.planning.handling import can_be_carried
from pivot.scene.graph import Scene, is_flat, is_opening


@dataclass(frozen=True)
class ParkingPockets:
    """Boundary proximity and remaining walking clearance, computed once per blocker."""

    floor: FloorMap
    distance: FloatArray
    nearest: IntArray
    walkable: BoolArray
    storage: BoolArray

    def accepts(self, box: OrientedBox) -> bool:
        """Requires sheltered parking with a free 60 cm pedestrian strip outside it.

        Args:
            box: Candidate final footprint, verified physically by the motion planner.

        Returns:
            Whether it is a storage pocket rather than a new mid-aisle obstacle.
        """
        corners = np.floor((box.footprint()-self.floor.origin)/self.floor.resolution).astype(int)
        corners[:,0] = np.clip(corners[:,0],0,self.floor.shape[1]-1)
        corners[:,1] = np.clip(corners[:,1],0,self.floor.shape[0]-1)
        if self.distance[corners[:,1],corners[:,0]].min() > .15:
            return False
        if not (self.floor.rasterize(box.footprint()) & ~self.storage).any():
            # Something fully under an existing footprint takes no new aisle area.
            # Its height, placement and departure/arrival still need motion validation.
            return True
        col,row = np.floor((box.center[:2]-self.floor.origin)/self.floor.resolution).astype(int)
        if not (0<=row<self.floor.shape[0] and 0<=col<self.floor.shape[1]):
            return False
        anchor = self.floor.origin + self.floor.resolution*(
            self.nearest[:,row,col][::-1]+.5)
        direction = box.center[:2]-anchor
        if np.linalg.norm(direction)<1e-9:
            return False
        direction /= np.linalg.norm(direction)
        extent = (abs(np.dot(direction,[np.cos(box.yaw),np.sin(box.yaw)]))*box.size[0]
                  + abs(np.dot(direction,[-np.sin(box.yaw),np.cos(box.yaw)]))*box.size[1])/2
        point = box.center[:2]+direction*(extent+WALKING_WIDTH/2
                                         + np.sqrt(2)*self.floor.resolution)
        col,row = np.floor((point-self.floor.origin)/self.floor.resolution).astype(int)
        return bool(0<=row<self.floor.shape[0] and 0<=col<self.floor.shape[1]
                    and self.walkable[row,col])


def parking_pockets(scene: Scene, object_id: int) -> ParkingPockets | None:
    """Builds geometric parking rules independent of whether a door starts blocked.

    Args:
        scene: Current complete layout.
        object_id: Blocker removed for the local destination analysis.

    Returns:
        Storage pockets with an outward pedestrian clearance test, or no floor data.
        This is not an evacuation-code or complete accessibility certification.
    """
    floor = scene.floor
    if floor is None:
        return None
    storage = np.zeros(floor.shape,dtype=bool)
    occupied = np.zeros(floor.shape,dtype=bool)
    for obj in scene.objects.values():
        if obj.object_id == object_id or is_flat(obj) or is_opening(obj):
            continue
        if not can_be_carried(obj) and obj.box.z_min<.1:
            storage |= floor.rasterize(obj.box.footprint())
        if obj.box.z_min<HEADROOM:
            occupied |= obstacle_mask(floor,obj)
    anchors = storage | ~floor.floor
    if floor.walls is not None:
        anchors |= floor.walls
    distance,nearest = ndimage.distance_transform_edt(
        ~np.pad(anchors,1,constant_values=True),return_indices=True)
    free = floor.floor & ~occupied
    walkable: BoolArray = (ndimage.distance_transform_edt(np.pad(free,1))[1:-1,1:-1]
                          * floor.resolution >= WALKING_WIDTH/2)
    return ParkingPockets(floor,distance[1:-1,1:-1]*floor.resolution,
                          nearest[:,1:-1,1:-1]-1,walkable,storage)
