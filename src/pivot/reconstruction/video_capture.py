"""A capture from any phone video: cameras and depth estimated by MapAnything.

ScanNet++ gives each frame its camera pose and a LiDAR depth map. A video
filmed with any phone (iPhone or Android, no LiDAR, no poses) gets them from
Meta's MapAnything, which regresses metric depth, intrinsics and camera
poses from images alone, in meters. Frames are picked evenly along the video
and written in the layout of a ScanNet++ iPhone capture
(:mod:`pivot.data.scannetpp`), with the video SAM 3.1 tracks objects in:
frame ``i`` of ``video.mp4`` is image ``i`` of the capture. The rest of PIVOT
(segmentation, lifting, the agent) runs on it unchanged; the floor found in
the points levels the scene, as phone poses have no gravity.

Runs on a GPU, with MapAnything installed (``scripts/gpu/setup_mapanything_env.sh``)::

    python -m pivot.reconstruction.video_capture --video room.mov --out capture/ --frames 150
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pivot.data.video import count_frames, ffmpeg_executable, frames_to_mp4
from pivot.geometry.types import FloatArray

# Meta's model with the Apache 2.0 license (the CC-BY-NC one is facebook/map-anything).
MODEL = "facebook/map-anything-apache"
# Frames given to MapAnything: enough views of every wall, within one GPU's memory.
FRAMES = 150
# Pixels below this confidence percentile get no depth: they would lift as noise.
CONFIDENCE_PERCENTILE = 10.0
# Depths beyond this are not kept, in meters (a window's view, a long corridor).
MAX_DEPTH = 8.0


def extract_frames(video: Path, count: int, folder: Path) -> list[Path]:
    """Extracts frames spread evenly along a video, upright as the phone filmed them.

    Args:
        video: Any video ffmpeg reads (an iPhone ``.MOV``, an Android ``.mp4``).
        count: Number of frames.
        folder: Where the JPEG frames go.

    Returns:
        The frames, in order.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    folder.mkdir(parents=True, exist_ok=True)
    total = count_frames(video)
    step = max(1, total // count)
    # ffmpeg applies the phone's rotation flag by default; frame n is kept if n % step == 0.
    command = [ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(video),
               "-vf", f"select=not(mod(n\\,{step}))", "-vsync", "vfr", "-q:v", "2",
               str(folder / "frame_%06d.jpg")]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    return sorted(folder.glob("frame_*.jpg"))[:count]


def _numpy(value: Any) -> FloatArray:
    """Converts a model output (a torch tensor) to an array, dropping the batch axis.

    Args:
        value: A tensor or an array, with a leading batch axis of 1.

    Returns:
        The array.
    """
    array = value.detach().float().cpu().numpy() if hasattr(value, "detach") else value
    result: FloatArray = np.asarray(array, dtype=np.float64)[0]
    return result


def write_capture(predictions: list[dict[str, Any]], out: Path) -> int:
    """Writes MapAnything's predictions as a capture in the ScanNet++ layout.

    Args:
        predictions: One dictionary per view, with ``img_no_norm``,
            ``depth_z``, ``conf``, ``intrinsics`` and ``camera_poses``
            (OpenCV camera-to-world, meters).
        out: The capture's directory.

    Returns:
        The number of frames written.
    """
    iphone = out / "iphone"
    (iphone / "rgb").mkdir(parents=True, exist_ok=True)
    (iphone / "depth").mkdir(parents=True, exist_ok=True)
    metadata = {}
    for index, view in enumerate(predictions):
        name = f"frame_{index:06d}"
        pixels = _numpy(view["img_no_norm"])
        # Denormalized images come in [0, 1] (or already in [0, 255]).
        scale = 255.0 if pixels.max() <= 1.5 else 1.0
        image = np.clip(pixels * scale, 0, 255).astype(np.uint8)
        Image.fromarray(image).save(iphone / "rgb" / f"{name}.jpg", quality=92)
        depth = _numpy(view["depth_z"])[..., 0]
        confidence = _numpy(view["conf"])
        keep = (confidence >= np.percentile(confidence, CONFIDENCE_PERCENTILE)) \
            & (depth > 0) & (depth < MAX_DEPTH)
        millimeters = np.where(keep, np.round(1000.0 * depth), 0).astype(np.uint16)
        Image.fromarray(millimeters).save(iphone / "depth" / f"{name}.png")
        metadata[name] = {"intrinsic": _numpy(view["intrinsics"]).tolist(),
                          "aligned_pose": _numpy(view["camera_poses"]).tolist()}
    (iphone / "pose_intrinsic_imu.json").write_text(json.dumps(metadata), encoding="utf-8")
    return len(metadata)


def capture_video(out: Path) -> Path:
    """Encodes a capture's frames into the video SAM 3.1 tracks objects in.

    Args:
        out: The capture's directory.

    Returns:
        ``<out>/video.mp4``; frame ``i`` is image ``i`` of the capture.
    """
    frames = sorted((out / "iphone" / "rgb").glob("frame_*.jpg"))
    with Image.open(frames[0]) as first:
        width = first.width - first.width % 2
    return frames_to_mp4(frames, out / "video.mp4", fps=6.0, width=width)


def reconstruct(video: Path, out: Path, frames: int = FRAMES, model_name: str = MODEL) -> Path:
    """Estimates cameras and depth for a phone video, and writes the capture.

    Args:
        video: The video.
        out: The capture's directory.
        frames: Number of frames to keep.
        model_name: MapAnything checkpoint.

    Returns:
        The capture's video for SAM 3.1.
    """
    # pylint: disable=import-outside-toplevel,import-error
    import torch  # Only on the GPU machine.
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(model_name).to(device)
    with tempfile.TemporaryDirectory() as folder:
        picked = extract_frames(video, frames, Path(folder))
        views = load_images([str(p) for p in picked])
        predictions = model.infer(views, memory_efficient_inference=True, use_amp=True,
                                  amp_dtype="bf16", apply_mask=True, mask_edges=True)
    count = write_capture(predictions, out)
    print(f"{count} frames with cameras and metric depth -> {out}", flush=True)
    return capture_video(out)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=FRAMES)
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)
    if (args.out / "iphone" / "pose_intrinsic_imu.json").exists():
        print(f"already reconstructed: {args.out}")
        if not (args.out / "video.mp4").exists():
            capture_video(args.out)
        return
    shutil.rmtree(args.out / "iphone", ignore_errors=True)
    reconstruct(args.video, args.out, args.frames, args.model)


if __name__ == "__main__":
    main()
