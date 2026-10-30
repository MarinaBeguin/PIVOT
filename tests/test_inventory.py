"""Tests for the object inventory that chooses SAM 3.1's prompts."""

from types import SimpleNamespace
from typing import Any

import numpy as np

from pivot.meta.client import MetaConfig
from pivot.perception.inventory import ask_api, contact_sheet, parse_inventory


def test_parse_reads_the_json_array_and_normalizes_names() -> None:
    text = 'Here: ["Chairs?", "the Table", "trash can", "door", "Window", "table"]'
    assert parse_inventory(text) == ["chairs", "table", "trash can", "door", "window"]


def test_parse_drops_kinds_already_covered_by_a_generic_name() -> None:
    assert parse_inventory('["office chair", "chair", "desk lamp", "lamp"]') == ["chair", "lamp"]


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
