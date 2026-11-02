"""Tests of objects brought into the room from a photo."""

import base64
import io
import json
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from pivot.agent.tools import Session, call_tool
from pivot.meta.client import MetaConfig
from pivot.meta.photo import (PhotoObject, cut_out, fill_top_notches, nouns_to_try,
                              object_from_photo, parse_description)
from pivot.perception.detections import Detection
from pivot.scene.examples import office_scene
from pivot.server.app import Room, create_app


def jpeg(width: int = 40, height: int = 30) -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (width, height), (200, 120, 40)).save(out, format="JPEG")
    return out.getvalue()


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


def test_the_description_is_read_and_its_sizes_kept_sensible() -> None:
    text = ('Sure: {"label": "Armchair", "width_m": 0.8, "depth_m": 0.95, "height_m": 9, '
            '"look": "red"}')
    assert parse_description(text) == ("armchair", (0.95, 0.8, 4.0), "red")
    assert parse_description('{"label": null}') is None
    assert parse_description("no idea") is None


def test_the_object_is_cut_out_of_the_photo() -> None:
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:8, 5:15] = True
    picture = Image.open(io.BytesIO(cut_out(jpeg(), [Detection(1, 0, (10, 5, 30, 15), mask)])))
    assert picture.size == (20, 10) and picture.mode == "RGBA"
    alpha = np.asarray(picture)[..., 3]
    assert alpha[5, 10] == 255 and alpha[0, 0] == 0
    assert Image.open(io.BytesIO(cut_out(jpeg(), []))).size == (40, 30)


def test_a_notch_left_by_a_vase_is_filled_but_not_the_gap_between_legs() -> None:
    mask = np.zeros((10, 40), dtype=bool)
    mask[2:6, :] = True  # The top of a sideboard...
    mask[6:, :3] = mask[6:, -3:] = True  # ... on two legs.
    mask[2:4, 18:22] = False  # Where a vase stood on it.
    mask[2:5, 35:] = False  # The far end of the top, lower in perspective.
    filled = fill_top_notches(mask)
    assert filled[2:4, 18:22].all()
    assert not filled[6:, 3:-3].any() and not filled[2:5, 35:].any()


def test_muse_names_the_object_and_sam_cuts_it_out() -> None:
    sent: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        sent.append(kwargs)
        answer = '{"label": "armchair", "width_m": 0.8, "depth_m": 0.8, "height_m": 0.9}'
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=answer))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    asked: list[str] = []

    def segment(path: Any, noun: str) -> list[Detection]:
        asked.append(noun)
        assert path.suffix == ".jpg"
        return []

    seen = object_from_photo(MetaConfig(api_key="k"), jpeg(), segment, client)
    # Nothing found: simpler nouns are tried, and the photo itself is the picture.
    assert seen is not None and seen.label == "armchair"
    assert asked == ["armchair", "furniture", "object"]
    assert nouns_to_try("leather armchair") == ["leather armchair", "armchair", "furniture",
                                               "object"]
    assert sent[0]["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg")


def test_an_added_object_goes_where_it_fits_and_can_be_resized_or_undone() -> None:
    session = Session(office_scene())
    added = call(session, "add_object", label="armchair", width_m=0.8, depth_m=0.85,
                 height_m=0.9, x=1.5, y=2.5)  # Aimed at the desk: put beside it.
    assert added["added"] and added["shift_m"] > 0.3
    new = session.scene[added["object_id"]]
    assert new.movable and new.box.z_min == 0.0
    assert call(session, "resize_object", object_id=new.object_id, width_m=0.9, depth_m=0.9,
                height_m=0.95)["resized"]
    window = call(session, "resize_object", object_id=5, width_m=1, depth_m=1, height_m=1)
    assert not window["resized"] and "cannot be changed" in window["reason"]
    assert call(session, "undo")["undone"] and call(session, "undo")["undone"]
    assert new.object_id not in session.scene.objects
    huge = call(session, "add_object", label="sofa", width_m=5.9, depth_m=5.9, height_m=1)
    assert not huge["added"] and "No free spot" in huge["reason"]


def test_the_server_brings_in_the_object_of_a_photo() -> None:
    points, colors = np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8)
    room = Room("office", "Office", office_scene(), points, colors)
    seen = PhotoObject("armchair", (0.8, 0.8, 0.9), "red velvet", b"\x89PNGcutout")
    client = TestClient(create_app([room], photographer=lambda photo: seen))
    assert '"photos": true' in client.get("/").text
    photo = "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode()
    answer = client.post("/api/photo", json={"image": photo}).json()
    assert answer["added"] and answer["look"] == "red velvet"
    assert answer["picture"].startswith("data:image/png;base64,")
    assert answer["object_id"] in {o["id"] for o in answer["objects"]}
    assert "Then bring in: the armchair" in client.get("/instructions").text
    nothing = TestClient(create_app([room], photographer=lambda photo: None))
    assert nothing.post("/api/photo", json={"image": photo}).json() == {
        "added": False, "reason": "no_object"}
