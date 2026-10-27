"""Common output of every reconstruction backend.

A reconstruction is a set of posed RGB frames with a metric depth map each.
Ground-truth loaders (ScanNet++) and learned reconstructors produce the same
structure, so the rest of the pipeline is backend-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pivot.geometry.types import FloatArray

DepthArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Reconstruction:
    """Posed frames with metric depth.

    Camera conventions follow OpenCV: x right, y down, z forward.

    Attributes:
        image_paths: One RGB image per frame.
        image_size: ``(width, height)`` of the RGB images, in pixels.
        intrinsics: (N, 3, 3) pinhole intrinsics at RGB resolution.
        cam_to_world: (N, 4, 4) camera-to-world transforms, in meters.
        depths: (N, H, W) depth along the optical axis in meters; 0 marks
            invalid pixels. The depth resolution may differ from the RGB one.
    """

    image_paths: list[Path]
    image_size: tuple[int, int]
    intrinsics: FloatArray
    cam_to_world: FloatArray
    depths: DepthArray

    def __post_init__(self) -> None:
        n = len(self.image_paths)
        if not len(self.intrinsics) == len(self.cam_to_world) == len(self.depths) == n:
            raise ValueError("All per-frame arrays must have one entry per image.")

    def __len__(self) -> int:
        return len(self.image_paths)

    @property
    def depth_size(self) -> tuple[int, int]:
        """``(width, height)`` of the depth maps, in pixels."""
        return int(self.depths.shape[2]), int(self.depths.shape[1])

    def depth_intrinsics(self, index: int) -> FloatArray:
        """Returns the intrinsics of one frame rescaled to depth resolution.

        Args:
            index: Frame index.

        Returns:
            A (3, 3) intrinsics matrix.
        """
        scale_x = self.depth_size[0] / self.image_size[0]
        scale_y = self.depth_size[1] / self.image_size[1]
        result: FloatArray = np.diag([scale_x, scale_y, 1.0]) @ self.intrinsics[index]
        return result

    def up_direction(self) -> FloatArray:
        """Estimates the world up direction from the cameras.

        People film rooms holding the phone roughly upright, so the average
        camera "up" axis (-y in OpenCV) is a reliable sign reference.

        Returns:
            A (3,) unit vector.
        """
        up = -self.cam_to_world[:, :3, 1].mean(axis=0)
        result: FloatArray = up / np.linalg.norm(up)
        return result

    def transformed(self, transform: FloatArray) -> Reconstruction:
        """Applies a rigid world transform to all cameras.

        Args:
            transform: (4, 4) rigid transform from the current world frame.

        Returns:
            The reconstruction expressed in the new world frame.
        """
        return Reconstruction(self.image_paths, self.image_size, self.intrinsics,
                              transform @ self.cam_to_world, self.depths)
