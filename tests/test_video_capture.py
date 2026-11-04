"""Tests of captures made from any phone video, and of blurring what is private in them."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pivot.data.scannetpp import load_scannetpp_iphone
from pivot.perception.detections import Detection
from pivot.privacy import blur, blur_capture, private_masks
from pivot.reconstruction.video_capture import extract_frames, write_capture


def views(count: int = 3, width: int = 40, height: int = 30) -> list[dict[str, np.ndarray]]:
    """MapAnything-like predictions: a camera stepping along x, 2 m from a wall."""
    result = []
    for index in range(count):
        pose = np.eye(4)
        pose[0, 3] = 0.1 * index
        result.append({
            "img_no_norm": np.full((1, height, width, 3), 0.5),
            "depth_z": np.full((1, height, width, 1), 2.0),
            "conf": np.linspace(0.0, 1.0, height * width).reshape((1, height, width)),
            "intrinsics": np.array([[[30.0, 0, width / 2], [0, 30.0, height / 2], [0, 0, 1]]]),
            "camera_poses": pose[None]})
    return result


def test_a_capture_is_written_in_the_scannetpp_layout_and_read_back(tmp_path: Path) -> None:
    assert write_capture(views(), tmp_path) == 3
    recon = load_scannetpp_iphone(tmp_path, stride=1)
    assert len(recon) == 3 and recon.image_size == (40, 30)
    assert recon.cam_to_world[2, 0, 3] == pytest.approx(0.2)
    # Depth in meters; the least confident tenth of the pixels has none.
    assert recon.depths[0].max() == pytest.approx(2.0)
    assert (recon.depths[0] == 0).mean() == pytest.approx(0.1, abs=0.01)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_frames_are_spread_along_the_video(tmp_path: Path) -> None:
    from pivot.data.video import frames_to_mp4  # pylint: disable=import-outside-toplevel
    frames = []
    for i in range(30):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((24, 32, 3), 8 * i, dtype=np.uint8)).save(path)
        frames.append(path)
    video = frames_to_mp4(frames, tmp_path / "phone.mp4", width=32)
    picked = extract_frames(video, 5, tmp_path / "picked")
    assert len(picked) == 5
    shades = [int(np.asarray(Image.open(p)).mean()) for p in picked]
    assert shades == sorted(shades) and shades[-1] - shades[0] > 150


def test_private_pixels_are_blurred_with_a_margin(tmp_path: Path) -> None:
    write_capture(views(count=2, width=40, height=30), tmp_path)
    frame = tmp_path / "iphone" / "rgb" / "frame_000001.jpg"
    # A name on a letter box: a sharp black and white pattern in frame 1.
    pattern = np.zeros((30, 40, 3), dtype=np.uint8)
    pattern[10:20, 10:30] = np.indices((10, 20)).sum(axis=0)[..., None] % 2 * 255
    Image.fromarray(pattern).save(frame)

    def segment(_video: Path, prompt: str) -> list[Detection]:
        if prompt != "text":
            return []
        return [Detection(1, 1, (10, 10, 30, 20), np.ones((10, 20), dtype=bool))]

    masks = private_masks(tmp_path / "video.mp4", (40, 30), segment)
    assert list(masks) == [1] and masks[1].sum() == 200
    assert blur_capture(tmp_path, segment) == 1
    after = np.asarray(Image.open(frame), dtype=float)
    assert after[10:20, 10:30].std() < 0.5 * pattern[10:20, 10:30].std()
    assert (tmp_path / "privacy.json").exists()
    untouched = blur(Image.fromarray(pattern), np.zeros((30, 40), dtype=bool))
    assert np.array_equal(np.asarray(untouched), pattern)
