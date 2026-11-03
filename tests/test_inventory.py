"""Tests for the object inventory that chooses SAM 3.1's prompts."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.lifting.scene_builder import build_scene
from pivot.meta.client import MetaConfig
from pivot.perception.inventory import (
    BUILT_IN_KINDS, INVENTORY_PROMPT, ask_api, ask_api_video, contact_sheet, parse_inventory,
)
from pivot.perception.tracks import Track
from synthetic import make_reconstruction, orbit_poses, render


def test_parse_reads_the_json_array_and_normalizes_names() -> None:
    text = 'Here: ["Chairs?", "the Table", "trash can", "door", "Window", "table"]'
    assert parse_inventory(text) == ["chairs", "table", "trash can", "door", "window"]


def test_parse_drops_kinds_already_covered_by_a_generic_name() -> None:
    assert parse_inventory('["office chair", "chair", "desk lamp", "lamp"]') == ["chair", "lamp"]


def test_fittings_are_asked_for_and_kept_beside_their_generic_name() -> None:
    assert all(f'"{kind}"' in INVENTORY_PROMPT for kind in BUILT_IN_KINDS)
    assert "hung on the walls" in INVENTORY_PROMPT
    assert parse_inventory('["counter", "kitchen counter", "wardrobe", "fitted wardrobe"]') == [
        "counter", "kitchen counter", "wardrobe", "fitted wardrobe"]


def test_a_fitting_is_built_fixed_and_a_chair_movable() -> None:
    chair = OrientedBox(np.array([0.4, -0.3, 0.45]), np.array([0.6, 0.5, 0.9]), 0.5)
    counter = OrientedBox(np.array([-0.6, 0.5, 0.45]), np.array([1.6, 0.6, 0.9]), -0.2)
    poses = orbit_poses()
    depths, ids = render(poses, [chair, counter])
    tracks = [Track(i, label, {f: ids[f] == i for f in range(len(ids)) if (ids[f] == i).any()})
              for i, label in ((1, "chair"), (2, "kitchen counter"))]
    scene = build_scene(make_reconstruction(poses, depths, np.eye(4)), tracks).scene
    assert {o.label: o.movable for o in scene.objects.values()} == {
        "chair": True, "kitchen counter": False}


def test_parse_falls_back_to_lines_and_commas_and_caps_the_list() -> None:
    assert parse_inventory("chair, sofa\nbed") == ["chair", "sofa", "bed"]
    assert parse_inventory('["a", "b", "c"]', limit=2) == ["a", "b"]
    assert not parse_inventory("[]")


def test_contact_sheet_tiles_row_by_row() -> None:
    tiles = [np.full((2, 3, 3), value, dtype=np.uint8) for value in (1, 2, 3, 4)]
    sheet = contact_sheet(tiles, columns=3)
    assert sheet.shape == (4, 9, 3)
    assert sheet[0, 0, 0] == 1 and sheet[0, 8, 0] == 3 and sheet[2, 0, 0] == 4
    assert sheet[3, 8, 0] == 0


def test_ask_api_sends_the_sheet_to_the_vision_model() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content='["chair"]'))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = MetaConfig(api_key="k", vision_model="muse-spark-1.3")
    assert ask_api(config, b"jpeg", 7, client) == '["chair"]'
    assert len(calls) == 1
    call = calls[0]
    assert call["model"] == "muse-spark-1.3" and call["reasoning_effort"] == "low"
    text, image = call["messages"][0]["content"]
    assert "at most 7 kinds" in text["text"]
    assert image["image_url"]["url"] == "data:image/jpeg;base64,anBlZw=="


def test_muse_spark_can_watch_the_whole_video(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    def upload(**kwargs: Any) -> SimpleNamespace:
        calls.append(("upload", kwargs))
        return SimpleNamespace(id="file-9")

    def respond(**kwargs: Any) -> SimpleNamespace:
        calls.append(("respond", kwargs))
        return SimpleNamespace(output_text='["sofa", "door"]')

    client = SimpleNamespace(files=SimpleNamespace(create=upload),
                             responses=SimpleNamespace(create=respond))
    video = tmp_path / "room.mp4"
    video.write_bytes(b"not really a video")
    answer = ask_api_video(MetaConfig(api_key="k"), video, 5, client)
    assert parse_inventory(answer) == ["sofa", "door"]
    content = calls[1][1]["input"][0]["content"]
    assert content[1] == {"type": "input_file", "file_id": "file-9"}
    assert content[0]["text"].startswith("This video walks through a home")
