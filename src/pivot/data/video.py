"""Video encoding for SAM 3.1: frame sequences to MP4, and faster subsampled videos.

    python -m pivot.data.video frames <video.mp4>
    python -m pivot.data.video subsample <video.mp4> <step> <out.mp4>
"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path


def ffmpeg_executable() -> str:
    """Finds ffmpeg: on the PATH, else the one bundled with ``imageio-ffmpeg``.

    Returns:
        The executable's path.

    Raises:
        RuntimeError: If neither is installed.
    """
    found = shutil.which("ffmpeg")
    if found is not None:
        return found
    try:
        # Optional dependency, with a static ffmpeg; for hosts without one.
        import imageio_ffmpeg  # type: ignore[import-not-found]  # pylint: disable=import-outside-toplevel,import-error
    except ImportError:
        raise RuntimeError("ffmpeg is required to encode videos: install it, or "
                           "`pip install imageio-ffmpeg`.") from None
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def frames_to_mp4(frames: list[Path], output: Path, fps: float = 6.0, width: int = 960) -> Path:
    """Encodes images into an H.264 MP4, one video frame per image.

    Video frame ``i`` is ``frames[i]``, so frame indices returned by SAM map
    directly back to the input list.

    Args:
        frames: Image files, in temporal order.
        output: Destination ``.mp4`` file.
        fps: Frame rate written in the container.
        width: Output width in pixels; the height keeps the aspect ratio.

    Returns:
        The output path.

    Raises:
        RuntimeError: If ffmpeg is not installed or fails.
    """
    ffmpeg = ffmpeg_executable()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        # ffmpeg's image2 demuxer needs consecutive names.
        for i, frame in enumerate(frames):
            (Path(tmp) / f"{i:06d}{frame.suffix.lower()}").symlink_to(frame.resolve())
        pattern = str(Path(tmp) / f"%06d{frames[0].suffix.lower()}")
        command = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", pattern,
                   "-vf", f"scale={width}:-2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                   "-crf", "20", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    return output


def subsample_video(video: Path, step: int, output: Path) -> Path:
    """Keeps one frame out of ``step``, as an H.264 MP4.

    Frame ``i`` of the output is frame ``step * i`` of the input: a video made
    from a capture with stride ``s`` becomes the video of stride ``step * s``.

    Args:
        video: Input video.
        step: Keep one frame out of ``step``.
        output: Destination ``.mp4`` file.

    Returns:
        The output path.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".part.mp4")
    # setpts renumbers the kept frames, so that no player drops or repeats any.
    command = [ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(video),
               "-vf", f"select=not(mod(n\\,{step})),setpts=N/FRAME_RATE/TB", "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(partial)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    partial.replace(output)
    return output


def count_frames(video: Path) -> int:
    """Counts the frames of a video by decoding it.

    Args:
        video: Video file.

    Returns:
        The number of frames.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    command = [ffmpeg_executable(), "-v", "error", "-i", str(video), "-map", "0:v:0",
               "-f", "null", "-progress", "pipe:1", "-"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    frames = [line.split("=", 1)[1] for line in completed.stdout.splitlines()
              if line.startswith("frame=")]
    if completed.returncode != 0 or not frames:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    return int(frames[-1])


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point, for the run scripts.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description="Video helpers for the run scripts.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("frames", help="Print the number of frames.").add_argument(
        "video", type=Path)
    subsample = commands.add_parser("subsample", help="Keep one frame out of STEP.")
    subsample.add_argument("video", type=Path)
    subsample.add_argument("step", type=int)
    subsample.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    if args.command == "frames":
        print(count_frames(args.video))
    else:
        subsample_video(args.video, args.step, args.output)


if __name__ == "__main__":
    main()
