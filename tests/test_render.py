"""Tests of realistic views of a planned room, drawn by Muse Image."""

import base64
from types import SimpleNamespace
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from pivot.meta.client import MetaConfig
from pivot.meta.image import photo_view, render_prompt
from pivot.scene.examples import office_scene
from pivot.server.app import Room, create_app
from pivot.server.limits import Budget, Meter, SpendLedger

PNG = b"\x89PNG\r\n\x1a\nfake"


def fake_client(answer: Any, calls: list[dict[str, Any]]) -> SimpleNamespace:
    def post(path: str, **kwargs: Any) -> SimpleNamespace:
        calls.append({"path": path, **kwargs})
        return SimpleNamespace(data=[answer])

    return SimpleNamespace(post=post)


def test_the_view_and_the_photos_of_new_objects_are_sent_to_muse_image() -> None:
    calls: list[dict[str, Any]] = []
    answer = SimpleNamespace(b64_json=base64.b64encode(b"photo").decode(), url=None)
    config = MetaConfig(api_key="k")
    image = photo_view(config, PNG, [("the new armchair (#201)", b"\xff\xd8jpeg")],
                       fake_client(answer, calls))
    assert image == b"photo"
    call = calls[0]
    assert call["path"] == "/images/edits" and call["body"]["model"] == "muse-image-1.0"
    # Numbered keys: the API refuses the SDK's "image[]".
    assert [(key, file[2]) for key, file in call["files"]] == [("image[0]", "image/png"),
                                                               ("image[1]", "image/jpeg")]
    assert "@Image2 shows the new armchair (#201)" in call["body"]["prompt"]
    assert "same viewpoint" in render_prompt([])


def test_an_image_given_by_address_is_downloaded() -> None:
    answer = SimpleNamespace(b64_json=None, url="https://example.org/view.png")
    fetched = SimpleNamespace(content=b"downloaded", raise_for_status=lambda: None)
    image = photo_view(MetaConfig(api_key="k"), PNG, client=fake_client(answer, []),
                       get=lambda url, timeout: fetched)
    assert image == b"downloaded"


def test_the_server_draws_the_view_and_charges_it() -> None:
    points, colors = np.zeros((1, 3)), np.zeros((1, 3), dtype=np.uint8)
    room = Room("office", "Office", office_scene(), points, colors)
    ledger = SpendLedger(Budget(total_usd=1.0, visitor_daily_usd=1.0, visitor_daily_requests=5))
    sent: list[bytes] = []

    def painter(view: bytes, _references: list[tuple[str, bytes]]) -> bytes:
        sent.append(view)
        return PNG

    client = TestClient(create_app([room], meter=Meter(ledger, "salt", image_price=0.1),
                                   painter=painter))
    assert '"render": true' in client.get("/").text
    view = "data:image/png;base64," + base64.b64encode(PNG).decode()
    answer = client.post("/api/render", json={"image": view}).json()
    assert answer["image"].startswith("data:image/png;base64,") and sent == [PNG]
    assert client.post("/api/render", json={"image": "not an image"}).status_code == 400
    assert client.get("/api/limits").json()["requests_left"] == 4
    off = TestClient(create_app([room]))
    assert off.post("/api/render", json={"image": view}).status_code == 404
