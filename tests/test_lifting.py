"""End-to-end tests of the lifting chain on exactly rendered synthetic views."""

import numpy as np
import pytest

from pivot.geometry.box import OrientedBox, fit_box, rotation_2d
from pivot.geometry.types import FloatArray
from pivot.lifting.backproject import backproject, depth_edges, resize_mask
from pivot.lifting.floor import estimate_ceiling, room_orientation
from pivot.lifting.parts import join_parts
from pivot.lifting.objects import (
    assign_by_votes, contained_tracks, distinct_pairs, duplicate_groups, largest_cluster,
    lift_track, split_overlaps, voxel_votes,
)
from pivot.lifting.scene_builder import build_scene, fit_object_box, settle
from pivot.perception.tracks import Track
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject
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
        fitted = built.scene[obj_id].box
        np.testing.assert_allclose(fitted.size, truth.size, atol=0.06)
        assert fitted.z_min == pytest.approx(0.0, abs=0.02)
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


def test_duplicate_groups_merges_overlapping_views_only() -> None:
    rng = np.random.default_rng(1)
    chair = rng.uniform(0, 0.5, (400, 3))
    seat_only = chair[chair[:, 2] > 0.3]  # A partial view of the same chair.
    neighbor = chair + np.array([0.6, 0.0, 0.0])  # Another chair next to it.
    groups = duplicate_groups([chair, neighbor, seat_only, np.zeros((0, 3))], radius=0.05)
    assert groups == [[0, 2], [1], [3]]


def test_settle_grounds_floating_furniture_and_stacks_the_rest() -> None:
    table_top = SceneObject(1, "table", box((2.0, 2.0, 0.73), (1.4, 0.7, 0.12)))
    monitor = SceneObject(2, "monitor", box((2.0, 2.1, 0.99), (0.5, 0.2, 0.46)))
    window = SceneObject(3, "window", box((2.0, 0.0, 1.6), (1.2, 0.1, 1.2)), movable=False)
    wall_shelf = SceneObject(4, "shelf", box((0.2, 2.0, 1.95), (0.6, 0.25, 0.08)))
    settled = settle([table_top, monitor, window, wall_shelf], tolerance=0.10, max_fall=1.0)
    table, screen, glass, shelf = settled[0], settled[1], settled[2], settled[3]
    assert table.box.z_min == pytest.approx(0.0) and table.box.z_max == pytest.approx(0.79)
    assert screen.box.z_min == pytest.approx(0.79) and screen.box.z_max == pytest.approx(1.22)
    assert glass is window and shelf is wall_shelf


def test_depth_edges_mark_silhouettes_only() -> None:
    depth = np.full((6, 6), 3.0)
    depth[2:4, 2:4] = 1.0  # A near object in front of a wall.
    edges = depth_edges(depth, max_jump=0.05)
    assert edges[2, 2] and edges[1, 1] and not edges[0, 5] and not edges[5, 0]
    # A plane seen at a grazing angle: steep but regular depth.
    slanted = np.tile(1.0 + 0.3 * np.arange(8.0), (6, 1))
    assert not depth_edges(slanted, max_jump=0.05).any()


def test_room_orientation_finds_rotated_walls() -> None:
    rng = np.random.default_rng(2)
    # Walls of a 4 x 3 m room, rotated by 20 degrees, plus clutter.
    t = rng.uniform(0, 1, 4000)
    walls = np.concatenate([np.column_stack([4 * t[:1000], np.zeros(1000)]),
                            np.column_stack([4 * t[1000:2000], np.full(1000, 3.0)]),
                            np.column_stack([np.zeros(1000), 3 * t[2000:3000]]),
                            np.column_stack([np.full(1000, 4.0), 3 * t[3000:]])])
    xy = np.vstack([walls, rng.uniform(0, 3, (800, 2))]) @ rotation_2d(np.radians(20)).T
    points = np.column_stack([xy, rng.uniform(0.6, 2.4, len(xy))])
    assert np.degrees(room_orientation(points)) == pytest.approx(20.0, abs=0.6)


def test_fit_object_box_ignores_strays_and_aligns_with_walls() -> None:
    rng = np.random.default_rng(3)
    truth = OrientedBox(np.array([1.0, 1.0, 0.3]), np.array([0.6, 0.4, 0.6]), np.radians(25))
    local = rng.uniform(-0.5, 0.5, (3000, 3)) * truth.size
    cloud = np.column_stack([local[:, :2] @ rotation_2d(truth.yaw).T + truth.center[:2],
                             local[:, 2] + truth.center[2]])
    strays = np.array([[2.0, 1.0, 0.3], [1.0, 1.8, 0.1]])  # Flying pixels on a wall.
    fitted = fit_object_box(np.vstack([cloud, strays]), room_yaw=np.radians(20))
    assert np.degrees(fitted.yaw) == pytest.approx(20.0)
    assert fitted.size[0] == pytest.approx(0.6, abs=0.06)
    assert fitted.size[1] == pytest.approx(0.4, abs=0.06)


def test_duplicate_groups_joins_parts_on_one_footprint() -> None:
    rng = np.random.default_rng(4)
    legs = rng.uniform([0, 0, 0], [0.5, 0.5, 0.4], (300, 3))
    seat = rng.uniform([0, 0, 0.45], [0.5, 0.5, 0.5], (300, 3))
    other = legs + np.array([0.8, 0.0, 0.0])
    boxes = [fit_box(c) for c in (legs, seat, other)]
    assert duplicate_groups([legs, seat, other], radius=0.03, boxes=boxes) == [[0, 1], [2]]
    assert duplicate_groups([legs, seat, other], radius=0.03) == [[0], [1], [2]]
    ledge = seat + np.array([0.0, 0.0, 1.0])  # Same footprint, but far above.
    assert duplicate_groups([legs, ledge], radius=0.03,
                            boxes=[fit_box(legs), fit_box(ledge)]) == [[0], [1]]


def test_duplicate_groups_never_merges_covisible_tracks() -> None:
    rng = np.random.default_rng(5)
    left_desk = rng.uniform([0, 0, 0.7], [1.5, 0.8, 0.76], (500, 3))
    right_desk = left_desk + np.array([1.5, 0.0, 0.0])
    # A track spanning both desks would chain them; SAM saw both desks in frame 3.
    bridge = rng.uniform([1.0, 0, 0.7], [2.0, 0.8, 0.76], (300, 3))
    clouds = [left_desk, right_desk, bridge]
    boxes = [fit_box(c) for c in clouds]
    assert duplicate_groups(clouds, 0.03, boxes=boxes) == [[0, 1, 2]]
    groups = duplicate_groups(clouds, 0.03, boxes=boxes, distinct={(0, 1)})
    assert [0] in groups or [1] in groups
    assert len(groups) == 2


def test_distinct_pairs_needs_disjoint_masks_in_a_shared_frame() -> None:
    left, right, whole = (np.zeros((4, 8), dtype=bool) for _ in range(3))
    left[:, :4], right[:, 4:], whole[:, :] = True, True, True
    tracks = [Track(1, "desk", {0: left, 1: left}), Track(2, "desk", {1: right}),
              Track(3, "desk", {1: whole}), Track(4, "desk", {5: right})]
    # Desks side by side are distinct; the whole desk contains both; track 4 is never co-visible.
    assert distinct_pairs(tracks) == {(0, 1)}


def test_touching_parts_of_one_chair_are_joined_but_not_two_chairs() -> None:
    rng = np.random.default_rng(6)
    seat = rng.uniform([0, 0, 0.0], [0.5, 0.45, 0.48], (400, 3))
    back = rng.uniform([0, 0.45, 0.48], [0.5, 0.5, 0.95], (200, 3))
    neighbor = seat + np.array([0.52, 0.0, 0.0])  # A second chair, touching the first.
    assert join_parts([seat, back, neighbor], "chair", fit_box, 0.06) == [[0, 1], [2]]
    assert join_parts([seat, back], "plant", fit_box, 0.06) == [[0], [1]]  # Unknown size.



def test_points_claimed_by_two_objects_go_to_the_one_more_tracks_support() -> None:
    rng = np.random.default_rng(8)
    near = rng.uniform([0, 0, 0.7], [1.4, 0.8, 0.76], (400, 3))
    far = near + np.array([1.4, 0.0, 0.0])
    row = np.vstack([near, far])  # One track covered the whole row of desks.
    near_alone = [near[rng.permutation(400)[:300]] for _ in range(3)]
    owned = assign_by_votes([[row, far], near_alone], radius=0.1)
    assert owned[0][:, 0].min() > 1.3  # The near desk's points left the far desk.
    assert len(owned[1]) == len(np.unique(np.vstack(near_alone), axis=0))


def test_a_part_segmented_under_another_noun_is_found() -> None:
    lamp = np.zeros((40, 40), dtype=bool)
    lamp[5:35, 18:22] = True
    lamp[5:8, 10:30] = True
    head = np.zeros((40, 40), dtype=bool)
    head[5:8, 12:28] = True
    desk = np.zeros((40, 40), dtype=bool)
    desk[30:40, 0:40] = True
    tracks = [Track(1, "lamp", {0: lamp, 1: lamp}), Track(2, "shelf", {0: head, 1: head}),
              Track(3, "desk", {0: desk})]
    assert contained_tracks(tracks) == {1}


def test_desks_pushed_together_are_split_at_the_middle_of_their_overlap() -> None:
    rng = np.random.default_rng(6)
    left = rng.uniform([0.0, 0, 0], [1.6, 0.8, 0.76], (800, 3))  # Spills 20 cm into the right one.
    right = rng.uniform([1.4, 0, 0], [3.0, 0.8, 0.76], (800, 3))
    patch = rng.uniform([0.5, 0.2, -0.02], [1.0, 0.6, 0.0], (100, 3))  # Flat: not a neighbor.
    split = split_overlaps([left, right, patch], fit_box)
    assert split[0][:, 0].max() <= 1.5 + 1e-9 and split[1][:, 0].min() > 1.5 - 0.05
    assert len(split[2]) == len(patch)


def _room_points(ceiling: float | None, rng: np.random.Generator) -> FloatArray:
    """Samples the walls of a 5 x 4 m room up to 3 m, and its ceiling plane if given."""
    n = 40000
    side = rng.integers(0, 4, n)
    t = rng.random(n)
    x = np.where(side < 2, 5.0 * t, np.where(side == 2, 0.0, 5.0))
    y = np.where(side < 2, np.where(side == 0, 0.0, 4.0), 4.0 * t)
    walls = np.column_stack([x, y, 3.0 * rng.random(n)])
    if ceiling is None:
        return walls
    plane = np.column_stack([5.0 * rng.random(n), 4.0 * rng.random(n),
                             ceiling + rng.normal(0.0, 0.01, n)])
    return np.vstack([walls, plane])


def test_ceiling_is_the_horizontal_layer_above_the_walls() -> None:
    rng = np.random.default_rng(0)
    assert estimate_ceiling(_room_points(2.8, rng)) == pytest.approx(2.75, abs=0.051)


def test_ceiling_is_unknown_when_the_camera_never_looked_up() -> None:
    assert estimate_ceiling(_room_points(None, np.random.default_rng(1))) is None
