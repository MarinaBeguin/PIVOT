"""Loader for ScanNet++ iPhone captures, used as ground truth during development.

Expected layout (ScanNet++ v2)::

    <scene>/iphone/rgb/frame_XXXXXX.jpg     1920 x 1440 RGB frames
    <scene>/iphone/depth/frame_XXXXXX.png   256 x 192 uint16 depth, millimeters
    <scene>/iphone/pose_intrinsic_imu.json  per-frame intrinsics and poses
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from pivot.reconstruction.types import Reconstruction

_IMAGE_SIZE = (1920, 1440)


def load_scannetpp_iphone(scene_dir: Path, stride: int = 10,
                          max_frames: int | None = None) -> Reconstruction:
    """Loads a ScanNet++ iPhone capture with its ground-truth poses and depth.

    Poses are the ``aligned_pose`` entries, which live in the z-up frame of the
    laser-scanner mesh.

    Args:
        scene_dir: Scene directory, containing ``iphone/``.
        stride: Keep one frame out of ``stride`` (the capture runs at 60 fps).
        max_frames: Optional cap on the number of frames kept.

    Returns:
        The reconstruction.

    Raises:
        FileNotFoundError: If the capture metadata is missing.
    """
    iphone = scene_dir / "iphone"
    metadata_path = iphone / "pose_intrinsic_imu.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"No ScanNet++ iPhone metadata at {metadata_path}.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = sorted(metadata)[::stride][:max_frames]

    depths = np.stack([
        np.asarray(Image.open(iphone / "depth" / f"{name}.png"), dtype=np.float32) / 1000.0
        for name in names
    ])
    return Reconstruction(
        image_paths=[iphone / "rgb" / f"{name}.jpg" for name in names],
        image_size=_IMAGE_SIZE,
        intrinsics=np.array([metadata[name]["intrinsic"] for name in names], dtype=np.float64),
        cam_to_world=np.array([metadata[name]["aligned_pose"] for name in names],
                              dtype=np.float64),
        depths=depths,
    )
