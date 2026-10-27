"""Analytic ray caster producing exact depth and instance masks for tests."""

from pathlib import Path

import numpy as np

from pivot.geometry.box import OrientedBox, rotation_2d
from pivot.reconstruction.types import Reconstruction

WIDTH, HEIGHT = 128, 96
INTRINSICS = np.array([[100.0, 0.0, 64.0], [0.0, 100.0, 48.0], [0.0, 0.0, 1.0]])


def look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Returns an OpenCV camera-to-world transform looking from eye to target."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = eye
    return pose


def _ray_box(origins: np.ndarray, dirs: np.ndarray, box: OrientedBox) -> np.ndarray:
    """Returns ray parameters of the first box hit, inf where missed."""
    rot = rotation_2d(box.yaw)
    o = origins - box.center
    o = np.column_stack([o[:, :2] @ rot, o[:, 2]])
    d = np.column_stack([dirs[:, :2] @ rot, dirs[:, 2]])
    half = 0.5 * box.size
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - o) / d, (half - o) / d
    t_near = np.nanmax(np.minimum(t1, t2), axis=1)
    t_far = np.nanmin(np.maximum(t1, t2), axis=1)
    return np.where((t_near <= t_far) & (t_near > 0), t_near, np.inf)


def render(poses: list[np.ndarray], boxes: list[OrientedBox],
           floor_size: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Renders depth and instance ids (0 = floor, -1 = nothing, i + 1 = box i)."""
    rows, cols = np.mgrid[0:HEIGHT, 0:WIDTH]
    pixels = np.column_stack([cols.ravel() + 0.5, rows.ravel() + 0.5, np.ones(rows.size)])
    rays_cam = pixels @ np.linalg.inv(INTRINSICS).T
    depths, ids = [], []
    for pose in poses:
        origins = np.broadcast_to(pose[:3, 3], rays_cam.shape)
        dirs = rays_cam @ pose[:3, :3].T
        hits = [np.where(dirs[:, 2] < 0, -origins[:, 2] / dirs[:, 2], np.inf)]
        on_floor = np.abs(origins[:, :2] + hits[0][:, None] * dirs[:, :2]).max(axis=1)
        hits[0] = np.where(on_floor <= floor_size / 2, hits[0], np.inf)
        hits += [_ray_box(origins, dirs, b) for b in boxes]
        stack = np.stack(hits)
        first = stack.argmin(axis=0)
        t = stack.min(axis=0)
        # Ray parameter equals depth because camera rays have z = 1.
        depths.append(np.where(np.isfinite(t), t, 0.0).reshape(HEIGHT, WIDTH))
        ids.append(np.where(np.isfinite(t), first, -1).reshape(HEIGHT, WIDTH))
    return np.stack(depths).astype(np.float32), np.stack(ids)


def orbit_poses(n: int = 12, radius: float = 2.6, height: float = 1.5) -> list[np.ndarray]:
    """Cameras on a circle around the origin, looking at its center."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [look_at(np.array([radius * np.cos(a), radius * np.sin(a), height]),
                    np.array([0.0, 0.0, 0.4])) for a in angles]


def make_reconstruction(poses: list[np.ndarray], depths: np.ndarray,
                        world_transform: np.ndarray | None = None) -> Reconstruction:
    """Wraps rendered views, optionally re-expressed in another world frame."""
    transform = np.eye(4) if world_transform is None else world_transform
    return Reconstruction(
        image_paths=[Path(f"view_{i}.png") for i in range(len(poses))],
        image_size=(WIDTH, HEIGHT),
        intrinsics=np.repeat(INTRINSICS[None], len(poses), axis=0),
        cam_to_world=np.stack([transform @ p for p in poses]),
        depths=depths,
    )
