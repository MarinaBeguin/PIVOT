"""Tests of the newer Meta models: web look-up of sizes, the open version, MMS-TTS."""

import json
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pivot.agent.tools import Session, call_tool
from pivot.meta.client import MetaConfig
from pivot.meta.products import (
    SEARCH_PRICE, LookUp, ProductSize, describe_page, is_public, look_up_size, parse_size,
)
from pivot.scene.examples import office_scene
from pivot.server.app import Room, create_app
from pivot.server.voice import wav_bytes


def test_a_size_needs_a_name_three_dimensions_and_a_source() -> None:
    text = ('Here it is: {"name": "KIVIK 3-seat sofa", "width_cm": 95, "depth_cm": 228, '
            '"height_cm": 83, "source": "https://www.ikea.com/kivik"}')
    assert parse_size(text) == ProductSize("KIVIK 3-seat sofa", (2.28, 0.95, 0.83),
                                           "https://www.ikea.com/kivik")
    assert parse_size('{"name": null}') is None
    assert parse_size('{"name": "x", "width_cm": 1, "depth_cm": 1, "height_cm": 1}') is None
    assert parse_size("no idea") is None


def test_the_look_up_asks_with_web_search_and_counts_what_it_costs() -> None:
    sent: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        sent.append(kwargs)
        return SimpleNamespace(
            output_text='{"name": "Stool", "width_cm": 40, "depth_cm": 40, "height_cm": 45, '
                        '"source": "https://shop.example/stool"}',
            output=[SimpleNamespace(type="web_search_call"), SimpleNamespace(type="message")],
            usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0))

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    result = look_up_size(MetaConfig(api_key="k"), "a stool", client=client)
    assert sent[0]["tools"] == [{"type": "web_search"}] and "a stool" in sent[0]["input"]
    assert result.product is not None and result.product.size == (0.4, 0.4, 0.45)
    assert result.cost == pytest.approx(1.25 + SEARCH_PRICE)


def test_a_link_is_read_as_a_preview_and_passed_on_with_it() -> None:
    page = ('<html><head><title>Console | Facebook Marketplace</title>'
            '<meta property="og:title" content="Konsolentisch">'
            '<meta property="og:description" content="B:100 H:80 T:30 &amp; Eiche"></head></html>')
    assert describe_page(page) == ("Konsolentisch | B:100 H:80 T:30 & Eiche | "
                                   "Console | Facebook Marketplace")
    sent: list[str] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        sent.append(kwargs["input"])
        return SimpleNamespace(output_text='{"name": null}', output=[], usage=None)

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    look_up_size(MetaConfig(api_key="k"), "", "https://shop.example/item/1", client=client,
                 reader=lambda url: describe_page(page))
    assert "https://shop.example/item/1" in sent[0] and "B:100 H:80 T:30" in sent[0]


def test_only_public_addresses_are_read() -> None:
    for url in ("http://127.0.0.1:8000/", "http://localhost/", "http://10.0.0.1/",
                "http://169.254.169.254/latest", "file:///etc/passwd", "ftp://example.com/"):
        assert not is_public(url), url


def test_the_agent_tool_quotes_the_source_and_bills_the_search() -> None:
    session = Session(office_scene())
    off = json.loads(call_tool(session, "look_up_product", '{"product": "KIVIK"}'))
    assert not off["found"] and "Meta API" in off["reason"]
    session.look_up = lambda product, url: LookUp(
        ProductSize("KIVIK", (2.28, 0.95, 0.83), "https://www.ikea.com/kivik"), 0.01)
    found = json.loads(call_tool(session, "look_up_product", '{"product": "KIVIK"}'))
    assert found == {"found": True, "name": "KIVIK", "size_m": [2.28, 0.95, 0.83],
                     "source": "https://www.ikea.com/kivik"}
    assert session.spent == 0.01
    assert "error" in json.loads(call_tool(session, "look_up_product", "{}"))


def test_the_open_version_never_sends_the_meta_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIVOT_BACKEND", "open")
    monkeypatch.setenv("META_API_KEY", "secret-meta-key")
    monkeypatch.delenv("PIVOT_OPEN_API_KEY", raising=False)
    config = MetaConfig.from_env()
    assert config.api_key == "local" and "meta.ai" not in config.base_url
    monkeypatch.setenv("PIVOT_OPEN_API_KEY", "provider-key")
    assert MetaConfig.from_env().api_key == "provider-key"


def test_answers_are_read_by_the_server_voice_where_it_has_one() -> None:
    def speaker(text: str, language: str) -> bytes | None:
        return wav_bytes(np.zeros(160), 16000) if language == "fr" and text else None

    room = Room("office", "Office", office_scene(), np.zeros((1, 3)),
                np.zeros((1, 3), dtype=np.uint8))
    client = TestClient(create_app([room], speaker=speaker))
    assert '"voices": ["en", "fr", "de", "es"]' in client.get("/").text
    audio = client.post("/api/speak", json={"text": "Bonjour", "language": "fr"})
    assert audio.status_code == 200 and audio.content[:4] == b"RIFF"
    assert client.post("/api/speak", json={"text": "Ciao", "language": "it"}).status_code == 404
