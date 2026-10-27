"""Lifting 2D object tracks to clean 3D point clouds and boxes."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pivot.geometry.types import FloatArray, IntArray
from pivot.lifting.backproject import masked_points
from pivot.perception.tracks import Track
from pivot.reconstruction.types import Reconstruction


def voxel_votes(points_per_view: list[FloatArray],
                voxel: float) -> tuple[FloatArray, IntArray]:
    """Counts in how many views each occupied voxel is observed.

    Args:
        points_per_view: One (N_i, 3) point cloud per view.
        voxel: Voxel side length, in meters.

    Returns:
        A tuple ``(centers, votes)`` with the (M, 3) centers of occupied voxels
        and the (M,) number of distinct views that hit each of them.
    """
    keys = [np.unique(np.floor(p / voxel).astype(np.int64), axis=0)
            for p in points_per_view if len(p)]
    if not keys:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    voxels, votes = np.unique(np.vstack(keys), axis=0, return_counts=True)
    return (voxels + 0.5) * voxel, votes


def largest_cluster(points: FloatArray, radius: float) -> FloatArray:
    """Keeps the largest connected component of a point cloud.

    Two points are connected when closer than ``radius``. Mask leaks onto the
    floor or a wall form separate islands that this step removes.

    Args:
        points: (N, 3) points.
        radius: Connection radius, in meters.

    Returns:
        The points of the largest component.
    """
    if len(points) < 2:
        return points
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(points), len(points)))
    _, labels = connected_components(graph, directed=False)
    result: FloatArray = points[labels == np.bincount(labels).argmax()]
    return result


def lift_track(recon: Reconstruction, track: Track, voxel: float = 0.03,
               min_views: int = 2, erode: int = 1) -> FloatArray:
    """Fuses a track's masks across views into a clean object point cloud.

    Voxels seen by fewer than ``min_views`` masks are dropped (one-off mask
    errors), then only the largest connected component is kept.

    Args:
        recon: The reconstruction the track's frame indices refer to.
        track: The object track.
        voxel: Fusion voxel size, in meters.
        min_views: Minimum number of views that must agree on a voxel. Reduced
            automatically for objects seen in few frames.
        erode: Mask erosion radius in depth pixels.

    Returns:
        (M, 3) voxel centers of the object; may be empty.
    """
    views = [masked_points(recon, frame, mask, erode) for frame, mask in track.masks.items()]
    centers, votes = voxel_votes(views, voxel)
    needed = min(min_views, max(1, len(views) // 2))
    return largest_cluster(centers[votes >= needed], radius=1.5 * voxel)
