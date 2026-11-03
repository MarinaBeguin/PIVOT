"""Tests of tracking a video in parallel pieces: frames kept, pieces, joined tracks."""

import shutil
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from pivot.data.video import count_frames, frame_runs, frames_to_mp4, select_frames
from pivot.meta.client import MetaConfig
from pivot.meta.sam import (
    Decoding, SamClient, parse_video_block, parse_video_response, piece_videos, response_blocks,
    track_in_parallel,
)
from pivot.perception.chunks import join_chunks, mask_iou, plan_chunks
from pivot.perception.cotracker import inside, mask_points, point_linker
from pivot.perception.detections import Detection
from pivot.perception.keyframes import select_keyframes
from pivot.perception.sam3_local import load_detections, prompt_filename


def square(object_id: int, frame: int, left: int, top: int, size: int = 10) -> Detection:
    return Detection(object_id, frame, (left, top, left + size, top + size),
                     np.ones((size, size), dtype=bool), 0.9)


def test_pieces_overlap_and_a_short_tail_joins_the_piece_before() -> None:
    chunks = plan_chunks(list(range(700)), 300, overlap=8)
    assert [(c[0], c[-1]) for c in chunks] == [(0, 299), (292, 591), (584, 699)]
    assert plan_chunks(list(range(330)), 300) == [list(range(330))]
    assert len(plan_chunks(list(range(40)), 300)) == 1


def test_masks_overlap_by_intersection_over_union() -> None:
    assert mask_iou(square(1, 0, 0, 0), square(2, 0, 0, 0)) == 1.0
    assert mask_iou(square(1, 0, 0, 0), square(2, 0, 5, 0)) == pytest.approx(50 / 150)
    assert mask_iou(square(1, 0, 0, 0), square(2, 0, 20, 0)) == 0.0


def test_tracks_of_two_pieces_are_joined_where_the_pieces_overlap() -> None:
    chunks = [[0, 1, 2, 3], [2, 3, 4, 5]]
    first = [square(7, f, 10 * f, 0) for f in range(4)] + [square(8, 0, 0, 50)]
    # The second piece numbers its objects its own way; object 7 is now 1.
    second = [square(1, f, 10 * (f + 2), 0) for f in range(4)] + [square(2, 3, 80, 80)]
    joined = join_chunks([first, second], chunks)
    tracks: dict[int, list[int]] = {}
    for d in joined:
        tracks.setdefault(d.object_id, []).append(d.frame)
    assert sorted(map(sorted, tracks.values())) == [[0], [0, 1, 2, 3, 4, 5], [5]]
    assert len(joined) == 8  # Shared frames are not counted twice.


def test_a_track_hidden_at_the_cut_is_joined_by_following_its_pixels() -> None:
    chunks = [[0, 1, 2, 3], [2, 3, 4, 5]]
    first, second = [square(1, 0, 0, 0)], [square(4, 3, 30, 0)]
    apart = join_chunks([first, second], chunks)
    assert len({d.object_id for d in apart}) == 2
    calls = []

    def linker(last: Detection, start: Detection) -> float:
        calls.append((last.frame, start.frame))
        return 0.8

    joined = join_chunks([first, second], chunks, linker=linker)
    assert calls == [(0, 5)] and len({d.object_id for d in joined}) == 1


def test_the_point_linker_counts_followed_points_landing_in_the_new_mask(tmp_path: Path) -> None:
    last, first = square(1, 2, 0, 0), square(2, 4, 30, 0)
    points = mask_points(last, 16)
    assert len(points) == 16 and inside(last, points).all()

    def tracker(frames: Any, start: Any) -> tuple[Any, Any]:
        assert len(frames) == 3  # Frames 2 to 4.
        moved = start + np.array([30.0, 0.0])
        return np.stack([start, start, moved]), np.ones((3, len(start)), dtype=bool)

    reads = []

    def read(_video: Path, a: int, b: int, width: int, height: int) -> Any:
        reads.append((a, b))
        return np.zeros((b - a + 1, height, width, 3), dtype=np.uint8)

    link = point_linker(tmp_path / "v.mp4", (64, 32), tracker, read=read)
    assert link(last, first) == 1.0 and reads == [(2, 4)]


def test_frames_are_kept_once_the_camera_has_moved_or_turned() -> None:
    poses = np.repeat(np.eye(4)[None], 6, axis=0)
    poses[1, 0, 3] = 0.03  # Moved 3 cm: dropped.
    poses[2, 0, 3] = 0.10  # 10 cm from frame 0: kept.
    angle = np.radians(6.0)
    poses[3, :3, :3] = [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0],
                        [0, 0, 1]]
    poses[3, 0, 3] = 0.10  # Turned 6 degrees: kept.
    poses[4] = poses[3]  # Still: dropped, but the last frame is always kept.
    poses[5] = poses[3]
    assert select_keyframes(poses) == [0, 2, 3, 5]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_selected_frames_come_out_in_order(tmp_path: Path) -> None:
    frames = []
    for i in range(12):
        path = tmp_path / f"{i}.png"
        Image.fromarray(np.full((32, 48, 3), 20 * i, dtype=np.uint8)).save(path)
        frames.append(path)
    video = frames_to_mp4(frames, tmp_path / "all.mp4", width=48)
    assert frame_runs([0, 1, 2, 5, 7, 8]) == [(0, 2), (5, 5), (7, 8)]
    kept = select_frames(video, [0, 1, 5, 9], tmp_path / "kept.mp4")
    assert count_frames(kept) == 4
    # A cached piece with the wrong number of frames (a faulty ffmpeg) is made again.
    pieces = piece_videos(video, [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9]], tmp_path / "pieces")
    assert [count_frames(p) for p in pieces] == [6, 6]
    pieces[0].write_bytes(video.read_bytes())  # 12 frames instead of 6.
    again = piece_videos(video, [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9]], tmp_path / "pieces")
    assert again == pieces and count_frames(again[0]) == 6


def test_every_prompt_and_piece_runs_at_once_and_pieces_are_joined(tmp_path: Path) -> None:
    chunks = [[0, 1, 2, 3], [2, 3, 4, 5]]
    pieces = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    running, peak, lock = [0], [0], threading.Lock()
    barrier = threading.Barrier(4)

    def segment(piece: Path, _prompt: str) -> list[Detection]:
        with lock:
            running[0] += 1
            peak[0] = max(peak[0], running[0])
        barrier.wait(timeout=5)  # Waits until the four requests run together.
        with lock:
            running[0] -= 1
        start = 0 if piece == pieces[0] else 2
        return [square(1, f, 10 * (f + start), 0) for f in range(4)]

    track_in_parallel(["chair", "table"], tmp_path, segment, pieces, chunks, workers=8)
    assert peak[0] == 4
    for prompt in ("chair", "table"):
        detections = load_detections(tmp_path / prompt_filename(prompt))
        assert sorted(d.frame for d in detections) == [0, 1, 2, 3, 4, 5]
        assert {d.object_id for d in detections} == {1}
        assert not (tmp_path / prompt_filename(prompt)).with_suffix(".progress").exists()


def test_a_video_is_uploaded_once_for_every_prompt(tmp_path: Path) -> None:
    uploads, lock = [], threading.Lock()

    def create(**kwargs: Any) -> SimpleNamespace:
        with lock:
            uploads.append(kwargs)
        return SimpleNamespace(id="file-1")

    fake = SimpleNamespace(files=SimpleNamespace(create=create))
    video = tmp_path / "room.mp4"
    video.write_bytes(b"not really a video")
    config = MetaConfig(api_key="test")
    sam = SamClient(config, tmp_path / "cache", client=fake)  # type: ignore[arg-type]
    ids = {sam._upload(video) for _ in range(3)}  # pylint: disable=protected-access
    assert ids == {"file-1"} and len(uploads) == 1


def test_responses_are_decoded_in_processes_block_by_block(tmp_path: Path) -> None:
    # A real response: six frames of a chair, one line per frame.
    response = Path(__file__).parent / "data" / "sam_video_chair.txt"
    whole = parse_video_response(response)
    blocks = response_blocks(response, size=1)
    assert len(blocks) == 6
    assert [(d.frame, d.object_id) for b in blocks for d in parse_video_block(b)] == \
        [(d.frame, d.object_id) for d in whole]
    pieces = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    chunks = [list(range(0, 6)), list(range(6, 12))]
    track_in_parallel(["chair"], tmp_path, _unused, pieces, chunks, workers=2,
                      decoding=Decoding(_Same(response), workers=2))
    detections = load_detections(tmp_path / prompt_filename("chair"))
    assert sorted(d.frame for d in detections) == list(range(12))


class _Same:
    """Fetches the same response for every piece; picklable."""

    def __init__(self, response: Path):
        self.response = response

    def __call__(self, _piece: Path, _prompt: str) -> Path:
        return self.response


def _unused(_piece: Path, _prompt: str) -> list[Detection]:
    raise AssertionError("With decoding, responses are fetched, not segmented.")
