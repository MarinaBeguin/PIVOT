"""Video encoding for SAM 3.1: frame sequences to MP4, and faster subsampled videos.

    python -m pivot.data.video frames <video.mp4>
    python -m pivot.data.video subsample <video.mp4> <step> <out.mp4>
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt


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
        import imageio_ffmpeg  # pylint: disable=import-outside-toplevel,import-error
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


def subsample_video(video: Path, step: int, output: Path, width: int | None = None) -> Path:
    """Keeps one frame out of ``step``, as an H.264 MP4.

    Frame ``i`` of the output is frame ``step * i`` of the input: a video made
    from a capture with stride ``s`` becomes the video of stride ``step * s``.

    Args:
        video: Input video.
        step: Keep one frame out of ``step``.
        output: Destination ``.mp4`` file.
        width: Width of the output in pixels, the height keeping the aspect
            ratio; None keeps the input's size.

    Returns:
        The output path.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".part.mp4")
    # The kept frames are renumbered at the input's rate, read here: ffmpeg's own FRAME_RATE
    # is wrong on some versions, which then repeat frames.
    rate = video_rate(video)
    command = [ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(video),
               "-vf", f"select=not(mod(n\\,{step})),setpts=N/({rate}*TB)"
               + (f",scale={width}:-2" if width else ""), "-r", rate, "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(partial)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    partial.replace(output)
    return output


def frame_runs(indices: list[int]) -> list[tuple[int, int]]:
    """Groups sorted frame indices into runs of consecutive frames.

    Args:
        indices: Increasing frame indices.

    Returns:
        ``(first, last)`` of each run, both included.
    """
    runs: list[tuple[int, int]] = []
    for index in indices:
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def select_frames(video: Path, indices: list[int], output: Path) -> Path:
    """Keeps the given frames of a video, in order, at the same size.

    Frame ``i`` of the output is frame ``indices[i]`` of the input, so the
    frames of detections on the output map back through ``indices``.

    Args:
        video: Input video.
        indices: Increasing frame indices to keep.
        output: Destination ``.mp4`` file.

    Returns:
        The output path.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".part.mp4")
    keep = "+".join(f"between(n\\,{first}\\,{last})" for first, last in frame_runs(indices))
    with tempfile.TemporaryDirectory() as tmp:
        # The filter can be long (one term per run of frames): read from a file.
        script = Path(tmp) / "filter.txt"
        rate = video_rate(video)
        script.write_text(f"select={keep},setpts=N/({rate}*TB)", encoding="utf-8")
        command = [ffmpeg_executable(), "-y", "-loglevel", "error", "-i", str(video),
                   "-filter_script:v", str(script), "-r", rate, "-an",
                   "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(partial)]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    partial.replace(output)
    return output


def read_frames(video: Path, first: int, last: int, width: int, height: int
                ) -> npt.NDArray[np.uint8]:
    """Decodes a range of frames of a video.

    Args:
        video: Video file.
        first: First frame index.
        last: Last frame index, included.
        width: Frame width, in pixels.
        height: Frame height, in pixels.

    Returns:
        (last - first + 1, height, width, 3) RGB frames.

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    rate = video_rate(video)
    command = [ffmpeg_executable(), "-v", "error", "-i", str(video),
               "-vf", f"select=between(n\\,{first}\\,{last}),setpts=N/({rate}*TB)",
               "-r", rate, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    completed = subprocess.run(command, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.decode().strip()}")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(-1, height, width, 3)


def video_size(video: Path) -> tuple[int, int]:
    """Reads the frame size of a video.

    Args:
        video: Video file.

    Returns:
        ``(width, height)`` in pixels.

    Raises:
        RuntimeError: If ffmpeg cannot read it.
    """
    # ffmpeg describes its input on stderr: "Stream #0:0: Video: h264 ..., 960x720, ...".
    probe = subprocess.run([ffmpeg_executable(), "-i", str(video)], capture_output=True,
                           text=True, check=False)
    match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", probe.stderr)
    if match is None:
        raise RuntimeError(f"Cannot read the size of {video}: {probe.stderr.strip()}")
    return int(match.group(1)), int(match.group(2))


def video_rate(video: Path) -> str:
    """Reads the frame rate of a video.

    Args:
        video: Video file.

    Returns:
        Frames per second, as ffmpeg writes it ("30", "29.97"); "30" if it gives none.
    """
    # "Stream #0:0: Video: h264 ..., 960x720, ..., 29.97 fps, 29.97 tbr, ...".
    probe = subprocess.run([ffmpeg_executable(), "-i", str(video)], capture_output=True,
                           text=True, check=False)
    line = next((line for line in probe.stderr.splitlines() if "Video:" in line), "")
    match = re.search(r"(\d+(?:\.\d+)?) fps", line) or re.search(r"(\d+(?:\.\d+)?) tbr", line)
    return match.group(1) if match else "30"


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
    subsample.add_argument("--width", type=int, help="Scale to this width, in pixels.")
    commands.add_parser("width", help="Print the width in pixels.").add_argument(
        "video", type=Path)
    args = parser.parse_args(argv)
    if args.command == "frames":
        print(count_frames(args.video))
    elif args.command == "width":
        print(video_size(args.video)[0])
    else:
        subsample_video(args.video, args.step, args.output, args.width)


if __name__ == "__main__":
    main()
