"""Tests for video encoding and subsampling (need ffmpeg on the PATH)."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pivot.data.video import count_frames, frames_to_mp4, subsample_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_subsampling_keeps_one_frame_out_of_step(tmp_path: Path) -> None:
    frames = []
    for i in range(10):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((32, 48, 3), 20 * i, dtype=np.uint8)).save(path)
        frames.append(path)
    video = frames_to_mp4(frames, tmp_path / "all.mp4", width=48)
    assert count_frames(video) == 10
    every3 = subsample_video(video, 3, tmp_path / "every3.mp4")
    assert count_frames(every3) == 4  # Frames 0, 3, 6 and 9.
