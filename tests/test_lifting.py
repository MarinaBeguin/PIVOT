"""End-to-end tests of the lifting chain on exactly rendered synthetic views."""

import numpy as np
import pytest

from pivot.geometry.box import OrientedBox
from pivot.lifting.backproject import backproject, resize_mask
from pivot.lifting.objects import largest_cluster, lift_track, voxel_votes
from pivot.lifting.scene_builder import build_scene
from pivot.perception.tracks import Track
from synthetic import INTRINSICS, look_at, make_reconstruction, orbit_poses, render

CHAIR = OrientedBox(np.array([0.4, -0.3, 0.45]), np.array([0.6, 0.5, 0.9]), 0.5)
TABLE = OrientedBox(np.array([-0.6, 0.5, 0.375]), np.array([1.2, 0.7, 0.75]), -0.2)


def random_rigid(seed: int) -> np.ndarray:
    """Returns a random rigid transform with a large rotation."""
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    transform = np.eye(4)
    transform[:3, :3] = q
    transform[:3, 3] = rng.uniform(-3, 3, 3)
    return transform


def tracks_from_ids(ids: np.ndarray, labels: list[str]) -> list[Track]:
    """Converts rendered instance ids to tracks (box i has id i + 1)."""
    return [Track(i + 1, label, {f: ids[f] == i + 1 for f in range(len(ids))
                                 if (ids[f] == i + 1).any()})
            for i, label in enumerate(labels)]


def test_backproject_recovers_floor_point() -> None:
    pose = look_at(np.array([0.0, -2.0, 1.0]), np.array([0.0, 0.0, 0.0]))
    depth, _ = render([pose], [])
    points = backproject(depth[0].astype(np.float64), INTRINSICS, pose)
    # Depth maps are float32.
    np.testing.assert_allclose(points[:, 2], 0.0, atol=1e-6)


def test_resize_mask_nearest() -> None:
    mask = np.zeros((4, 4), dtype=bool)
    mask[:2, :2] = True
    np.testing.assert_array_equal(resize_mask(mask, (2, 2)), [[True, False], [False, False]])


def test_voxel_votes_count_distinct_views() -> None:
    point = np.array([[0.01, 0.01, 0.01]])
    centers, votes = voxel_votes([point, point, point + 1.0], voxel=0.1)
    assert sorted(votes.tolist()) == [1, 2]
    assert len(centers) == 2


def test_largest_cluster_drops_islands() -> None:
    rng = np.random.default_rng(0)
    blob = rng.uniform(0, 0.3, (300, 3))
    island = rng.uniform(2, 2.05, (10, 3))
    kept = largest_cluster(np.vstack([blob, island]), radius=0.1)
    assert len(kept) == 300


def test_lift_track_recovers_box() -> None:
    poses = orbit_poses()
    depths, ids = render(poses, [CHAIR])
    track = tracks_from_ids(ids, ["chair"])[0]
    cloud = lift_track(make_reconstruction(poses, depths), track, voxel=0.02, erode=0)
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    corners = CHAIR.corners()
    np.testing.assert_allclose(lo, corners.min(axis=0), atol=0.05)
    np.testing.assert_allclose(hi, corners.max(axis=0), atol=0.05)


@pytest.mark.parametrize("seed", [0, 1])
def test_build_scene_levels_and_fits_boxes(seed: int) -> None:
    poses = orbit_poses()
    depths, ids = render(poses, [CHAIR, TABLE])
    # Express everything in an arbitrary frame, as a learned reconstructor would.
    recon = make_reconstruction(poses, depths, random_rigid(seed))
    built = build_scene(recon, tracks_from_ids(ids, ["chair", "table"]))

    assert sorted(o.label for o in built.scene.objects.values()) == ["chair", "table"]
    for truth, obj_id in ((CHAIR, 1), (TABLE, 2)):
        box = built.scene[obj_id].box
        np.testing.assert_allclose(box.size, truth.size, atol=0.06)
        assert box.z_min == pytest.approx(0.0, abs=0.02)
    # The leveled frame differs from the original by a rotation about z only,
    # so object distances are preserved.
    gap = np.linalg.norm(built.scene[1].box.center[:2] - built.scene[2].box.center[:2])
    assert gap == pytest.approx(np.linalg.norm(CHAIR.center[:2] - TABLE.center[:2]), abs=0.05)


def test_build_scene_floor_covers_furniture_footprints() -> None:
    poses = orbit_poses()
    depths, ids = render(poses, [CHAIR, TABLE])
    tracks = tracks_from_ids(ids, ["chair", "table"])
    built = build_scene(make_reconstruction(poses, depths), tracks)
    assert built.scene.floor is not None
    table = built.scene[2].box
    assert built.scene.floor.floor_fraction(table.footprint()) > 0.95


def test_leveled_cameras_keep_height() -> None:
    poses = orbit_poses(height=1.5)
    depths, ids = render(poses, [CHAIR])
    recon = make_reconstruction(poses, depths, random_rigid(3))
    built = build_scene(recon, tracks_from_ids(ids, ["chair"]))
    np.testing.assert_allclose(built.recon.cam_to_world[:, 2, 3], 1.5, atol=0.01)
