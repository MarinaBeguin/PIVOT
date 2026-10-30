"""Small synthetic scenes with known geometry, for tests, demos and evaluation."""

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.floor import FloorMap
from pivot.geometry.shape import VoxelShape
from pivot.scene.graph import Scene, SceneObject


def box(center: tuple[float, float, float], size: tuple[float, float, float],
        yaw: float = 0.0) -> OrientedBox:
    """Builds a box from plain tuples.

    Args:
        center: Box center, in meters.
        size: Full extents, in meters.
        yaw: Rotation about the vertical axis, in radians.

    Returns:
        The box.
    """
    return OrientedBox(np.array(center, dtype=np.float64), np.array(size, dtype=np.float64), yaw)


def rectangle_floor(width: float, depth: float, resolution: float = 0.05) -> FloorMap:
    """Builds the floor map of an empty rectangular room with a corner at the origin.

    Args:
        width: Extent along x, in meters.
        depth: Extent along y, in meters.
        resolution: Floor map cell size, in meters.

    Returns:
        The floor map.
    """
    xs, ys = np.meshgrid(np.arange(0.0, width, 0.02), np.arange(0.0, depth, 0.02))
    return FloorMap.from_points(np.column_stack([xs.ravel(), ys.ravel()]), resolution,
                                close_radius=0)


def office_scene() -> Scene:
    """A 5 x 4 m office: desk with a lamp on it, chair, shelf and a window.

    Layout (top view, meters)::

        y=4 +-------------[window]---------+
            |                    shelf     |
            |   desk+lamp                  |
            |   chair                      |
        y=0 +------------------------------+
           x=0                            x=5

    Returns:
        The scene.
    """
    objects = [
        SceneObject(1, "desk", box((1.5, 2.5, 0.375), (1.4, 0.7, 0.75))),
        SceneObject(2, "lamp", box((1.2, 2.6, 0.95), (0.2, 0.2, 0.4))),
        SceneObject(3, "office chair", box((1.5, 1.7, 0.5), (0.6, 0.6, 1.0))),
        SceneObject(4, "shelf", box((4.0, 3.8, 1.0), (1.0, 0.4, 2.0))),
        SceneObject(5, "window", box((2.8, 3.95, 1.6), (1.2, 0.1, 1.2)), movable=False),
    ]
    return Scene.from_objects(objects, rectangle_floor(5.0, 4.0), {"name": "synthetic office"})


def two_rooms_floor(door_width: float) -> FloorMap:
    """Two 4 x 4 m rooms side by side, separated by a 10 cm wall with a door.

    The wall spans x in [4.0, 4.1]; the door is centered at y = 2.

    Args:
        door_width: Width of the door opening, in meters.

    Returns:
        The floor map.
    """
    xs, ys = np.meshgrid(np.arange(0.0, 8.1, 0.02), np.arange(0.0, 4.0, 0.02))
    points = np.column_stack([xs.ravel(), ys.ravel()])
    in_wall = (points[:, 0] > 3.99) & (points[:, 0] < 4.11)
    in_door = np.abs(points[:, 1] - 2.0) < door_width / 2
    return FloorMap.from_points(points[~in_wall | in_door], resolution=0.05, close_radius=0)


def sofa_scene(door_width: float) -> Scene:
    """A 2.0 x 0.9 m sofa standing lengthwise along y in the left of two rooms.

    Args:
        door_width: Width of the door between the rooms, in meters.

    Returns:
        The scene.
    """
    sofa = SceneObject(1, "sofa", box((2.0, 2.0, 0.375), (2.0, 0.9, 0.75), yaw=np.pi / 2))
    return Scene.from_objects([sofa], two_rooms_floor(door_width),
                              {"name": f"two rooms, {door_width:.2f} m door"})


def legged_table(object_id: int, center_xy: tuple[float, float],
                 size: tuple[float, float, float] = (1.2, 0.8, 0.76), top: float = 0.08,
                 leg: float = 0.08, resolution: float = 0.04) -> SceneObject:
    """A table with a voxel shape: a top on four square legs, free underneath.

    Args:
        object_id: Object id.
        center_xy: Center of the table on the floor, in meters.
        size: Full extents, in meters.
        top: Thickness of the top, in meters.
        leg: Side of the square legs, in meters.
        resolution: Cell side length, in meters.

    Returns:
        The table.
    """
    table_box = box((center_xy[0], center_xy[1], 0.5 * size[2]), size)
    half = np.array(size) / 2
    axes = [np.arange(-h, h, resolution) + 0.5 * resolution for h in half]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    in_top = grid[:, 2] >= half[2] - top
    in_leg = np.all(np.abs(grid[:, :2]) >= half[:2] - leg, axis=1)
    cells = np.floor(grid[in_top | in_leg] / resolution).astype(np.int64)
    return SceneObject(object_id, "table", table_box, shape=VoxelShape(resolution, cells))
