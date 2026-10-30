"""Colored point clouds of a reconstruction, for visualization."""

import numpy as np
import numpy.typing as npt
from PIL import Image

from pivot.geometry.types import FloatArray
from pivot.lifting.backproject import backproject
from pivot.reconstruction.types import Reconstruction

ColorArray = npt.NDArray[np.uint8]


def colored_points(recon: Reconstruction, frame_stride: int = 4, pixel_stride: int = 3,
                   voxel: float = 0.02, max_depth: float = 6.0) -> tuple[FloatArray, ColorArray]:
    """Fuses colored depth pixels into a voxel-downsampled point cloud.

    Args:
        recon: The reconstruction.
        frame_stride: Use one frame out of ``frame_stride``.
        pixel_stride: Use one depth pixel out of ``pixel_stride`` along each axis.
        voxel: Output voxel size, in meters; one point (mean color) per voxel.
        max_depth: Depths beyond this are discarded.

    Returns:
        A tuple ``(points, colors)``: (N, 3) world points and (N, 3) RGB colors.
    """
    width, height = recon.depth_size
    grid = np.zeros((height, width), dtype=bool)
    grid[::pixel_stride, ::pixel_stride] = True
    clouds, colors = [], []
    for frame in range(0, len(recon), frame_stride):
        depth = recon.depths[frame]
        valid = grid & (depth > 0) & (depth < max_depth)
        image = Image.open(recon.image_paths[frame]).convert("RGB").resize((width, height))
        clouds.append(backproject(depth, recon.depth_intrinsics(frame),
                                  recon.cam_to_world[frame], valid, max_depth))
        colors.append(np.asarray(image)[valid])
    means = voxel_average(np.vstack(clouds), np.vstack(colors).astype(np.float64), voxel)
    return means[:, :3], np.clip(means[:, 3:], 0, 255).astype(np.uint8)


def voxel_average(points: FloatArray, values: FloatArray, voxel: float) -> FloatArray:
    """Averages points and their attributes per voxel.

    Args:
        points: (N, 3) points.
        values: (N, C) per-point attributes.
        voxel: Voxel size, in meters.

    Returns:
        (M, 3 + C) mean position and attributes of each occupied voxel.
    """
    keys = np.floor(points / voxel).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3 + values.shape[1]))
    np.add.at(sums, inverse.ravel(), np.hstack([points, values]))
    result: FloatArray = sums / counts[:, None]
    return result
