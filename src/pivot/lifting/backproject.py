"""Back-projection of depth pixels to world points."""

import numpy as np
from scipy import ndimage

from pivot.geometry.types import BoolArray, FloatArray
from pivot.reconstruction.types import Reconstruction


def backproject(depth: FloatArray, intrinsics: FloatArray, cam_to_world: FloatArray,
                mask: BoolArray | None = None, max_depth: float = 8.0) -> FloatArray:
    """Lifts valid depth pixels to world coordinates.

    Args:
        depth: (H, W) depth along the optical axis, in meters; 0 is invalid.
        intrinsics: (3, 3) intrinsics at depth resolution.
        cam_to_world: (4, 4) camera-to-world transform.
        mask: Optional (H, W) mask restricting the pixels to lift.
        max_depth: Depths beyond this are discarded as unreliable.

    Returns:
        (N, 3) world points.
    """
    valid = (depth > 0) & (depth < max_depth)
    if mask is not None:
        valid &= mask
    rows, cols = np.nonzero(valid)
    z = depth[rows, cols].astype(np.float64)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    # Pixel centers sit at integer + 0.5 in the OpenCV convention used by ScanNet++.
    camera = np.column_stack([(cols + 0.5 - cx) / fx * z, (rows + 0.5 - cy) / fy * z, z])
    result: FloatArray = camera @ cam_to_world[:3, :3].T + cam_to_world[:3, 3]
    return result


def resize_mask(mask: BoolArray, size: tuple[int, int]) -> BoolArray:
    """Resizes a mask with nearest-neighbor sampling.

    Args:
        mask: (H, W) mask.
        size: Target ``(width, height)``.

    Returns:
        The resized (height, width) mask.
    """
    width, height = size
    if mask.shape == (height, width):
        return mask
    rows = ((np.arange(height) + 0.5) * mask.shape[0] / height).astype(int)
    cols = ((np.arange(width) + 0.5) * mask.shape[1] / width).astype(int)
    result: BoolArray = mask[rows[:, None], cols[None, :]]
    return result


def depth_edges(depth: FloatArray, max_jump: float = 0.05) -> BoolArray:
    """Finds depth pixels lying on a depth discontinuity.

    At an object's silhouette a low-resolution depth pixel averages the object
    and what is behind it, producing "flying" points between the two. A plane
    seen at a grazing angle also has large depth differences between
    neighbors, but they are regular: discontinuities are detected on the
    second difference (curvature) along rows and columns instead, then grown
    by one pixel to catch the flying pixel between the two sides of a step.

    Args:
        depth: (H, W) depth map, in meters.
        max_jump: Largest second difference, relative to depth, that is not a
            discontinuity.

    Returns:
        An (H, W) mask, True on discontinuities.
    """
    # Second differences are only defined inside the image; borders stay 0.
    curvature = np.zeros_like(depth)
    curvature[:, 1:-1] = np.abs(depth[:, :-2] + depth[:, 2:] - 2 * depth[:, 1:-1])
    curvature[1:-1, :] = np.maximum(curvature[1:-1, :],
                                    np.abs(depth[:-2, :] + depth[2:, :] - 2 * depth[1:-1, :]))
    result: BoolArray = ndimage.binary_dilation(curvature > max_jump * depth)
    return result


def masked_points(recon: Reconstruction, frame: int, mask: BoolArray,
                  erode: int = 1, max_depth: float = 8.0,
                  max_jump: float | None = 0.05) -> FloatArray:
    """Lifts the pixels of one frame's mask to world points.

    Masks are eroded and depth discontinuities removed first: depth at object
    boundaries mixes foreground and background and would smear the object
    towards the wall behind it.

    Args:
        recon: The reconstruction.
        frame: Frame index.
        mask: Mask at any resolution.
        erode: Erosion radius in depth pixels.
        max_depth: Depths beyond this are discarded.
        max_jump: Relative depth jump marking a discontinuity, see
            :func:`depth_edges`; None keeps every pixel.

    Returns:
        (N, 3) world points.
    """
    small = resize_mask(mask, recon.depth_size)
    if erode > 0:
        small = ndimage.binary_erosion(small, iterations=erode)
    depth = recon.depths[frame]
    if max_jump is not None:
        small &= ~depth_edges(depth, max_jump)
    return backproject(depth, recon.depth_intrinsics(frame), recon.cam_to_world[frame], small,
                       max_depth)


def scene_points(recon: Reconstruction, pixel_stride: int = 4,
                 max_depth: float = 8.0) -> FloatArray:
    """Lifts a subsample of all depth pixels, for whole-scene analysis.

    Args:
        recon: The reconstruction.
        pixel_stride: Keep one pixel out of ``pixel_stride`` along each axis.
        max_depth: Depths beyond this are discarded.

    Returns:
        (N, 3) world points.
    """
    width, height = recon.depth_size
    grid = np.zeros((height, width), dtype=bool)
    grid[::pixel_stride, ::pixel_stride] = True
    clouds = [backproject(recon.depths[frame], recon.depth_intrinsics(frame),
                          recon.cam_to_world[frame], grid, max_depth)
              for frame in range(len(recon))]
    return np.vstack(clouds)
