"""Frames worth tracking: those where the camera has moved since the last one kept.

SAM 3.1 is billed, and takes time, per frame and per kind of object. A phone
held still, or panned slowly, gives many frames that show nothing new: a frame
is kept only once the camera has moved by ``min_move`` meters or turned by
``min_turn_deg`` degrees since the last kept frame. No view is lost: every
dropped frame is that close to a kept one. On a slowly walked apartment, half
the frames go; on a quickly scanned office, a few percent::

    python -m pivot.perception.keyframes --scene <scannetpp_scene> --stride 10 --out keyframes.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from pivot.geometry.types import FloatArray

# Defaults measured on the ScanNet++ captures at 6 frames per second: the
# longest run of dropped frames stays under about a second and a half.
MIN_MOVE = 0.08
MIN_TURN_DEG = 5.0


def select_keyframes(cam_to_world: FloatArray, min_move: float = MIN_MOVE,
                     min_turn_deg: float = MIN_TURN_DEG) -> list[int]:
    """Picks the frames where the camera has moved or turned enough.

    Args:
        cam_to_world: (N, 4, 4) camera poses, in temporal order.
        min_move: Distance from the last kept camera that keeps a frame, in meters.
        min_turn_deg: Rotation from the last kept camera that keeps a frame, in degrees.

    Returns:
        Increasing frame indices; the first and last frames are always kept.
    """
    if len(cam_to_world) == 0:
        return []
    kept = [0]
    cos_turn = math.cos(math.radians(min_turn_deg))
    for i in range(1, len(cam_to_world)):
        last, pose = cam_to_world[kept[-1]], cam_to_world[i]
        moved = float(np.linalg.norm(pose[:3, 3] - last[:3, 3]))
        relative = last[:3, :3].T @ pose[:3, :3]
        # cos of the rotation angle, from the trace of the relative rotation.
        cos_angle = 0.5 * (float(np.trace(relative)) - 1.0)
        if moved >= min_move or cos_angle <= cos_turn:
            kept.append(i)
    if kept[-1] != len(cam_to_world) - 1:
        kept.append(len(cam_to_world) - 1)
    return kept


def scannetpp_poses(scene_dir: Path, stride: int) -> FloatArray:
    """Reads the camera poses of a ScanNet++ iPhone capture, without its images.

    Args:
        scene_dir: Scene directory, containing ``iphone/``.
        stride: Keep one frame out of ``stride``, as the video was made.

    Returns:
        (N, 4, 4) camera-to-world poses.
    """
    metadata = json.loads((scene_dir / "iphone" / "pose_intrinsic_imu.json")
                          .read_text(encoding="utf-8"))
    names = sorted(metadata)[::stride]
    return np.array([metadata[name]["aligned_pose"] for name in names], dtype=np.float64)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: writes the kept frame indices as JSON.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description="Frames where the camera moved enough.")
    parser.add_argument("--scene", type=Path, required=True, help="ScanNet++ scene directory.")
    parser.add_argument("--stride", type=int, default=10, help="Frame stride of the video.")
    parser.add_argument("--min-move", type=float, default=MIN_MOVE)
    parser.add_argument("--min-turn", type=float, default=MIN_TURN_DEG)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    poses = scannetpp_poses(args.scene, args.stride)
    kept = select_keyframes(poses, args.min_move, args.min_turn)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(kept), encoding="utf-8")
    print(f"{len(kept)} of {len(poses)} frames kept ({100 * len(kept) / max(len(poses), 1):.0f}%)",
          flush=True)


if __name__ == "__main__":
    main()
