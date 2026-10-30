"""Builds a PIVOT scene from a ScanNet++ capture and SAM 3.1 detections.

The detections come from :mod:`pivot.perception.sam3_local` (one ``.npz`` per
prompt), computed on the capture's video made with the same frame stride, so
video frame ``i`` is reconstruction frame ``i``.

Usage::

    python -m pivot.build --scene data/scannetpp/b20a261fdf \\
        --detections detections/b20a261fdf_s10 --out outputs/b20a261fdf
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np

from pivot.data.scannetpp import load_scannetpp_iphone
from pivot.lifting.scene_builder import BuildConfig, BuiltScene, build_scene
from pivot.perception.detections import tracks_from_detections
from pivot.perception.sam3_local import load_detections, prompt_filename
from pivot.perception.tracks import Track
from pivot.reconstruction.types import Reconstruction
from pivot.viz.pointcloud import colored_points
from pivot.viz.web import export_viewer


def label_from_filename(path: Path) -> str:
    """Recovers the prompt of a detection file, e.g. ``trash_can.npz`` -> ``trash can``.

    Args:
        path: Detection file written by :mod:`pivot.perception.sam3_local`.

    Returns:
        The prompt.
    """
    return path.stem.replace("_", " ")


def load_tracks(detection_dir: Path, video_width: int, image_size: tuple[int, int],
                prompts: list[str] | None = None) -> list[Track]:
    """Loads detection files of a directory as tracks with unique ids.

    Args:
        detection_dir: Directory of ``.npz`` detection files.
        video_width: Width of the video SAM ran on; its height follows the
            aspect ratio of the capture.
        image_size: Capture image size ``(width, height)``.
        prompts: Prompts to load; every file of the directory if None. A
            directory caching the detections of a video may hold prompts of
            earlier object lists, which would duplicate objects.

    Returns:
        Tracks of the prompts, ids numbered from 1.
    """
    height = round(video_width * image_size[1] / image_size[0])
    paths = (sorted(detection_dir.glob("*.npz")) if prompts is None
             else [detection_dir / prompt_filename(prompt) for prompt in prompts])
    tracks: list[Track] = []
    for path in paths:
        tracks += tracks_from_detections(load_detections(path), label_from_filename(path),
                                         video_width, height, first_id=len(tracks) + 1)
    return tracks


def summarize(built: BuiltScene, tracks: list[Track]) -> str:
    """Describes the lifted objects, one per line.

    Args:
        built: The built scene.
        tracks: The tracks it was built from.

    Returns:
        A table of objects with their size, position and support.
    """
    lines = [f"{len(built.scene.objects)} objects kept out of {len(tracks)} tracks"]
    for obj in sorted(built.scene.objects.values(), key=lambda o: o.object_id):
        box = obj.box
        lines.append(
            f"  {obj.object_id:3d} {obj.label:10s} size {box.size.round(2)} "
            f"center {box.center.round(2)} yaw {math.degrees(box.yaw):6.1f} deg "
            f"z [{box.z_min:.2f}, {box.z_max:.2f}] "
            f"{len(built.object_points[obj.object_id])} voxels")
    return "\n".join(lines)


def build(recon: Reconstruction, detection_dir: Path, out_dir: Path,
          video_width: int = 960, prompts: list[str] | None = None) -> BuiltScene:
    """Builds the scene, then writes ``scene.json``, ``cloud.npz`` and ``viewer.html``.

    Args:
        recon: The capture.
        detection_dir: Directory of ``.npz`` detection files.
        out_dir: Output directory.
        video_width: Width of the video SAM ran on.
        prompts: Prompts to load; every detection file if None.

    Returns:
        The built scene.
    """
    tracks = load_tracks(detection_dir, video_width, recon.image_size, prompts)
    built = build_scene(recon, tracks, BuildConfig())
    out_dir.mkdir(parents=True, exist_ok=True)
    built.scene.save(out_dir / "scene.json")
    points, colors = colored_points(built.recon)
    # The server reloads the cloud with the scene, without the capture.
    np.savez_compressed(out_dir / "cloud.npz", points=points.astype(np.float32), colors=colors)
    export_viewer(built.scene, points, colors, out_dir / "viewer.html",
                  title=f"PIVOT - {detection_dir.name}")
    print(summarize(built, tracks), flush=True)
    return built


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--scene", type=Path, required=True, help="ScanNet++ scene directory.")
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=10,
                        help="Frame stride the video was made with.")
    parser.add_argument("--video-width", type=int, default=960)
    parser.add_argument("--prompts", nargs="+", default=None,
                        help="Prompts to load; every detection file if omitted.")
    args = parser.parse_args(argv)

    start = time.time()
    recon = load_scannetpp_iphone(args.scene, stride=args.stride)
    print(f"{len(recon)} frames loaded in {time.time() - start:.0f}s", flush=True)
    build(recon, args.detections, args.out, args.video_width, args.prompts)
    print(f"Done in {time.time() - start:.0f}s: {args.out}", flush=True)


if __name__ == "__main__":
    main()
