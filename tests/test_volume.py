"""Tests of volume carving on exactly rendered synthetic views."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pivot.geometry.box import OrientedBox
from pivot.geometry.shape import VoxelShape
from pivot.geometry.solid import solid_penetration
from pivot.lifting.completion import complete_hidden, drop_fragments
from pivot.lifting.scene_builder import build_scene
from pivot.lifting.volume import CarveConfig, carve_shapes, observation_counts, separate_shapes
from pivot.perception.tracks import Track
from pivot.scene.graph import SceneObject
from pivot.scene.examples import box, legged_table
from synthetic import make_reconstruction, orbit_poses, render

# A table: a top on four legs, and a closed carton next to it.
TABLE_PARTS = [box((0.0, 0.0, 0.725), (1.2, 0.7, 0.05))] + [
    box((x, y, 0.35), (0.06, 0.06, 0.7)) for x in (-0.55, 0.55) for y in (-0.3, 0.3)]
CARTON = box((1.0, 0.9, 0.25), (0.5, 0.5, 0.5))


def views() -> list[np.ndarray]:
    # High views see the top, low ones see under it.
    return orbit_poses(16, radius=2.2, height=1.6) + orbit_poses(16, radius=2.2, height=0.3)


def occupied(shape: VoxelShape, obj_box: OrientedBox, point: tuple[float, float, float]) -> bool:
    cell = np.floor(obj_box.to_local(np.array([point])) / shape.resolution).astype(np.int64)
    return bool((shape.cells == cell).all(axis=1).any())


def test_shape_follows_its_box() -> None:
    shape = VoxelShape(0.1, np.array([[0, 0, 0]]))
    moved = box((1.0, 2.0, 0.5), (1.0, 1.0, 1.0), yaw=np.pi / 2)
    np.testing.assert_allclose(shape.world_centers(moved), [[0.95, 2.05, 0.55]])
    assert VoxelShape.from_dict(shape.to_dict()).cells.tolist() == [[0, 0, 0]]


def test_points_behind_a_surface_are_not_free() -> None:
    poses = orbit_poses(4)
    depths, _ = render(poses, [CARTON])
    recon = make_reconstruction(poses, depths)
    seen = observation_counts(
        recon, np.array([[0.0, 0.0, 0.6], [1.0, 0.9, 0.25], [1.0, 0.9, 0.5]]), CarveConfig())
    assert seen.free[0] >= 2  # Open space above the floor.
    assert seen.free[1] == 0 and seen.surface[1] == 0  # Inside the carton: never observed.
    assert seen.surface[2] >= 1  # On its lid.


@pytest.fixture(name="built", scope="module")
def fixture_built() -> dict[str, tuple[VoxelShape, OrientedBox]]:
    poses = views()
    depths, ids = render(poses, [*TABLE_PARTS, CARTON])
    frames = range(len(poses))
    table = Track(1, "table", {f: (ids[f] >= 1) & (ids[f] <= 5) for f in frames})
    carton = Track(2, "carton", {f: ids[f] == 6 for f in frames})
    # Depth coarser than masks, as on phones: thin legs are then only found
    # through the masks.
    scene = build_scene(make_reconstruction(poses, depths[:, ::2, ::2]), [table, carton]).scene
    result = {}
    for obj in scene.objects.values():
        assert obj.shape is not None
        result[obj.label] = (obj.shape, obj.box)
    return result


def test_space_under_a_table_is_free(built: dict[str, tuple[VoxelShape, OrientedBox]]) -> None:
    shape, table_box = built["table"]
    assert occupied(shape, table_box, (0.0, 0.0, 0.72))  # The top.
    assert not occupied(shape, table_box, (0.0, 0.0, 0.35))  # Under it.
    assert not occupied(shape, table_box, (0.3, -0.15, 0.15))
    assert len(shape) < 0.5 * np.prod(table_box.size) / shape.resolution ** 3


def test_inside_of_a_closed_box_stays_occupied(
        built: dict[str, tuple[VoxelShape, OrientedBox]]) -> None:
    shape, carton_box = built["carton"]
    assert occupied(shape, carton_box, (1.0, 0.9, 0.25))
    assert len(shape) > 0.7 * np.prod(carton_box.size) / shape.resolution ** 3


def test_thin_legs_are_kept(built: dict[str, tuple[VoxelShape, OrientedBox]]) -> None:
    shape, table_box = built["table"]
    legs = [(x, y, z) for x in (-0.55, 0.55) for y in (-0.3, 0.3) for z in (0.2, 0.45)]
    assert sum(occupied(shape, table_box, p) for p in legs) >= 6


def test_carving_nothing_is_a_no_op() -> None:
    poses = orbit_poses(2)
    depths, _ = render(poses, [])
    assert not carve_shapes(make_reconstruction(poses, depths), [], {}, {})


def test_separate_shapes_removes_boundary_overlaps() -> None:
    cube = np.stack(np.meshgrid(*[np.arange(-5, 5)] * 3, indexing="ij"), -1).reshape(-1, 3)
    # Two 40 cm cubes whose grids overlap by about one cell where they touch.
    boxes = [box((0.0, 0.0, 0.2), (0.4, 0.4, 0.4)), box((0.37, 0.01, 0.2), (0.4, 0.4, 0.4))]
    objects = [SceneObject(i, "cube", b, shape=VoxelShape(0.04, cube)) for i, b in enumerate(boxes)]
    assert solid_penetration(boxes[0], objects[0].shape, boxes[1], objects[1].shape) > 0
    surfaces = [b.center + np.array([[-0.18, 0, 0], [0.18, 0, 0]]) for b in boxes]
    separated = separate_shapes(objects, [cKDTree(s) for s in surfaces])
    assert solid_penetration(boxes[0], separated[0].shape, boxes[1], separated[1].shape) == 0.0
    assert all(len(s.shape) >= 0.85 * len(cube) for s in separated if s.shape is not None)


def test_a_chair_half_under_a_desk_is_completed_under_it() -> None:
    desk = legged_table(1, (2.0, 2.0), size=(1.4, 0.8, 0.76))
    # Only the back of the chair was seen: 20 cm deep, sticking out of the desk's front edge.
    # Its backrest (y < 0, all heights) and the start of its seat (z from 0.40 to 0.48 m).
    back = box((2.0, 1.55, 0.45), (0.5, 0.2, 0.9))
    cells = np.stack(np.meshgrid(np.arange(-6, 6), np.arange(-2, 3), np.arange(-11, 11),
                                 indexing="ij"), -1).reshape(-1, 3)
    seat = (cells[:, 2] >= -1) & (cells[:, 2] <= 0)
    cells = cells[(cells[:, 1] < 0) | seat]
    chair = SceneObject(2, "chair", back, shape=VoxelShape(0.04, cells))
    completed = complete_hidden([desk, chair])
    grown = completed[1]
    assert grown.box.size[1] == pytest.approx(0.5)
    assert grown.box.center[1] > back.center[1]  # Towards the desk.
    centers = grown.shape.world_centers(grown.box)
    under_desk = centers[centers[:, 1] > 1.7]
    # The seat goes on under the desk, at its own height; the backrest does not.
    assert len(under_desk)
    assert under_desk[:, 2].min() > 0.39 and under_desk[:, 2].max() < 0.49
    # A cell of margin around the desk: the two volumes do not interlock.
    gaps, _ = cKDTree(desk.shape.world_centers(desk.box)).query(under_desk)
    assert gaps.min() > 0.04
    assert completed[0] is desk


def test_a_small_object_touching_nothing_is_left_alone() -> None:
    chair = SceneObject(2, "chair", box((1.0, 1.0, 0.45), (0.5, 0.2, 0.9)))
    assert complete_hidden([chair])[0] is chair


def test_a_fragment_next_to_an_object_of_its_label_is_dropped() -> None:
    chair = SceneObject(2, "chair", box((1.9, 2.1, 0.41), (0.7, 0.35, 0.82)))
    # The top of its backrest, seen from far away, 30 cm off but touching it.
    fragment = SceneObject(13, "chair", box((1.54, 1.85, 0.72), (0.44, 0.25, 0.15)))
    lone = SceneObject(20, "chair", box((4.0, 1.0, 0.72), (0.44, 0.25, 0.15)))
    kept = drop_fragments([chair, fragment, lone])
    assert [o.object_id for o in kept] == [2, 20]
