"""Lifting 2D object tracks to clean 3D point clouds and boxes."""

import itertools
from collections.abc import Callable

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import polygon_penetration
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


def duplicate_groups(clouds: list[FloatArray], radius: float, min_overlap: float = 0.5,
                     boxes: list[OrientedBox | None] | None = None, max_gap: float = 0.2,
                     distinct: set[tuple[int, int]] | None = None) -> list[list[int]]:
    """Groups point clouds that describe the same physical object.

    SAM gives an object a new id when it re-detects it after losing it, so one
    object can produce several tracks. Two clouds are duplicates when most of
    the smaller one lies within ``radius`` of the other or, if boxes are
    given, above or below the other's footprint and vertically close to it:
    partial views of one chair (its seat, its legs) share few points but
    stand on the same footprint.

    Some pairs are known to be distinct objects (SAM separated them in a
    frame where both are visible). Pairs are merged from the most to the
    least similar, and a merge that would put a distinct pair in one group is
    refused, so that two desks standing side by side are never fused, not
    even through a third track.

    Args:
        clouds: Object point clouds.
        radius: Distance under which two points coincide, in meters.
        min_overlap: Fraction of the smaller cloud that must coincide.
        boxes: Optional box fitted to each cloud, None for empty clouds.
        max_gap: Largest vertical gap between parts on one footprint, in meters.
        distinct: Index pairs ``(i, j)``, ``i < j``, that must stay apart.

    Returns:
        Groups of cloud indices, each sorted, ordered by their first index.
    """
    trees = [cKDTree(c) if len(c) else None for c in clouds]
    scored = sorted(((score, i, j) for i in range(len(clouds)) for j in range(i + 1, len(clouds))
                     if (score := _similarity(clouds, trees, (i, j), radius, boxes, max_gap))
                     >= min_overlap), reverse=True)
    return _merge_greedily([(i, j) for _, i, j in scored], len(clouds), distinct or set())


def _merge_greedily(pairs: list[tuple[int, int]], count: int,
                    distinct: set[tuple[int, int]]) -> list[list[int]]:
    """Merges items pair by pair, refusing merges that would join a distinct pair.

    Args:
        pairs: Pairs to merge, most similar first.
        count: Number of items.
        distinct: Index pairs ``(i, j)``, ``i < j``, that must stay apart.

    Returns:
        Groups of item indices, each sorted, ordered by their first index.
    """
    group_of = list(range(count))
    members = {i: [i] for i in range(count)}
    for i, j in pairs:
        a, b = group_of[i], group_of[j]
        if a == b or any((min(x, y), max(x, y)) in distinct
                         for x in members[a] for y in members[b]):
            continue
        for index in members[b]:
            group_of[index] = a
        members[a] += members.pop(b)
    return sorted(sorted(group) for group in members.values())


def _similarity(clouds: list[FloatArray], trees: list[cKDTree | None], pair: tuple[int, int],
                radius: float, boxes: list[OrientedBox | None] | None, max_gap: float) -> float:
    """Measures how much of the smaller of two clouds lies on the larger one.

    Args:
        clouds: Point clouds.
        trees: Their KD-trees, None for empty clouds.
        pair: Indices of the two clouds.
        radius: Distance under which two points coincide, in meters.
        boxes: Optional box of each cloud; lying over its footprint, within
            ``max_gap`` of it vertically, also counts.
        max_gap: Largest vertical gap between parts on one footprint, in meters.

    Returns:
        The fraction of the smaller cloud, in [0, 1]; 0 if a cloud is empty.
    """
    i, j = pair
    small, big = (i, j) if len(clouds[i]) <= len(clouds[j]) else (j, i)
    tree = trees[big]
    if tree is None or trees[small] is None:
        return 0.0
    distances, _ = tree.query(clouds[small], distance_upper_bound=radius)
    score = float(np.mean(np.isfinite(distances)))
    box = boxes[big] if boxes is not None else None
    if box is not None:
        score = max(score, _over_footprint(clouds[small], box, radius, max_gap))
    return score


def _over_footprint(points: FloatArray, box: OrientedBox, margin: float, max_gap: float) -> float:
    """Returns the fraction of points standing over a box's footprint.

    Args:
        points: (N, 3) points.
        box: The box.
        margin: Growth of the footprint on every side, in meters.
        max_gap: Points vertically farther than this from the box count as
            none over it, in meters.

    Returns:
        The fraction in [0, 1].
    """
    gap = max(box.z_min - points[:, 2].max(), points[:, 2].min() - box.z_max)
    if gap > max_gap:
        return 0.0
    local = np.abs(box.to_local(points)[:, :2])
    return float(np.mean(np.all(local <= 0.5 * box.size[:2] + margin, axis=1)))


def distinct_pairs(tracks: list[Track], max_shared: float = 0.5, min_area: float = 0.005,
                   samples: int = 5) -> set[tuple[int, int]]:
    """Finds the pairs of tracks that SAM separated in a frame where both are visible.

    Two masks of the same frame belong to distinct objects when they barely
    overlap; a mask largely inside the other (a desk top and the whole desk)
    is a part of the same object. Small masks are fragments whose separation
    proves nothing.

    Args:
        tracks: Tracks of one label.
        max_shared: Masks sharing more than this fraction of the smaller one
            are parts of the same object.
        min_area: Masks covering less than this fraction of the frame are
            not evidence of distinct objects.
        samples: Number of shared frames examined per pair.

    Returns:
        Index pairs ``(i, j)``, ``i < j``, of distinct objects.
    """
    result = set()
    for i, a in enumerate(tracks):
        for j in range(i + 1, len(tracks)):
            b = tracks[j]
            shared = sorted(f for f in set(a.masks) & set(b.masks)
                            if min(a.masks[f].mean(), b.masks[f].mean()) >= min_area)
            if not shared:
                continue
            picked = [shared[k] for k in np.linspace(0, len(shared) - 1, min(samples, len(shared)),
                                                     dtype=int)]
            fractions = [np.logical_and(a.masks[f], b.masks[f]).sum()
                         / max(min(a.masks[f].sum(), b.masks[f].sum()), 1) for f in picked]
            if np.median(fractions) <= max_shared:
                result.add((i, j))
    return result



def assign_by_votes(track_clouds: list[list[FloatArray]], radius: float) -> list[FloatArray]:
    """Gives each point claimed by several objects to the one most of its tracks support.

    A track can cover two neighboring objects of its label (a whole row of
    desks) while other tracks cover each of them alone. A point belongs to the
    object with the most tracks having a point within ``radius`` of it; ties
    keep it in every tied object.

    Args:
        track_clouds: For each object, the point cloud of each of its tracks.
        radius: Distance under which a track supports a point, in meters.

    Returns:
        The point cloud of each object.
    """
    trees = [[cKDTree(c) for c in clouds if len(c)] for clouds in track_clouds]
    unions = [np.unique(np.vstack(clouds), axis=0) if any(len(c) for c in clouds)
              else np.zeros((0, 3)) for clouds in track_clouds]
    result = []
    for index, points in enumerate(unions):
        votes = np.stack([_support(points, object_trees, radius) for object_trees in trees])
        result.append(points[votes[index] >= votes.max(axis=0)])
    return result


def _support(points: FloatArray, trees: list[cKDTree], radius: float) -> IntArray:
    """Counts, for each point, the tracks having a point within a radius.

    Args:
        points: (N, 3) points.
        trees: KD-trees of the tracks' clouds.
        radius: Support distance, in meters.

    Returns:
        (N,) counts.
    """
    counts = np.zeros(len(points), dtype=np.int64)
    for tree in trees:
        distances, _ = tree.query(points, distance_upper_bound=radius)
        counts += np.isfinite(distances)
    return counts


def contained_tracks(tracks: list[Track], min_inside: float = 0.8) -> set[int]:
    """Finds the tracks that are parts of an object of another label.

    SAM may segment a part of an object under another noun: the flat head of
    a floor lamp as a "shelf", the glazing of a door as a "window". In every
    frame where both are seen, the part's mask then lies inside the whole's
    mask, so its points are already the whole's points.

    Args:
        tracks: Object tracks.
        min_inside: Median fraction of a track's mask inside the other's for
            it to be a part.

    Returns:
        Indices of the part tracks.
    """
    parts = set()
    for i, part in enumerate(tracks):
        for whole in tracks:
            shared = sorted(set(part.masks) & set(whole.masks))
            if whole.label == part.label or not shared:
                continue
            inside = [np.logical_and(part.masks[f], whole.masks[f]).sum()
                      / max(part.masks[f].sum(), 1) for f in shared]
            if np.median(inside) >= min_inside:
                parts.add(i)
                break
    return parts


def split_overlaps(clouds: list[FloatArray], fit: Callable[[FloatArray], OrientedBox]
                   ) -> list[FloatArray]:
    """Separates objects of one label whose boxes overlap, at the middle of the overlap.

    Two desks pushed together share an edge, and points of one spill into
    the other when tracks covered both. Along the line joining their centers,
    each object keeps its points up to the middle of the overlapping span.
    Only objects side by side are split: they must also share most of their
    height (a flat patch under a chair is not a neighbor).

    Args:
        clouds: Point clouds of distinct objects of one label.
        fit: Box fitting function.

    Returns:
        The clouds, trimmed.
    """
    clouds = list(clouds)
    for i, j in itertools.combinations(range(len(clouds)), 2):
        a, b = clouds[i], clouds[j]
        if min(len(a), len(b)) == 0:
            continue
        box_a, box_b = fit(a), fit(b)
        shared_height = min(box_a.z_max, box_b.z_max) - max(box_a.z_min, box_b.z_min)
        if (polygon_penetration(box_a.footprint(), box_b.footprint()) <= 0.0
                or shared_height < 0.5 * min(box_a.size[2], box_b.size[2])):
            continue
        axis = box_b.center[:2] - box_a.center[:2]
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        along_a, along_b = a[:, :2] @ axis, b[:, :2] @ axis
        low, high = np.quantile(along_b, 0.02), np.quantile(along_a, 0.98)
        if low >= high:
            continue
        cut = 0.5 * (low + high)
        clouds[i], clouds[j] = a[along_a <= cut], b[along_b > cut]
    return clouds
