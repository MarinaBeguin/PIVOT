"""Tests of the viewer's languages: its texts, and the agent answering in the page's language."""

import json
import re
from importlib import resources
from types import SimpleNamespace
from typing import Any

import numpy as np

from pivot.agent.loop import LANGUAGE_NAMES, Agent
from pivot.agent.tools import Session
from pivot.meta.client import MetaConfig
from pivot.scene.examples import office_scene
from pivot.viz.web import render_viewer


def _languages_script() -> str:
    return resources.files("pivot.viz").joinpath("i18n.js").read_text(encoding="utf-8")


def test_every_language_has_every_text() -> None:
    script = _languages_script()
    blocks = dict(re.findall(r"\n  (\w\w): \{\n(.*?)\n  \},", script, re.DOTALL))
    keys = {code: set(re.findall(r'(?:^\s+|, )(\w+): "', block, re.MULTILINE))
            for code, block in blocks.items()}
    assert set(keys) == {"en", "fr", "de", "es", "it", "zh"} == set(LANGUAGE_NAMES)
    for code, found in keys.items():
        # Only the catch-all note for unknown warnings may fall back to English.
        assert keys["en"] - found <= {"note"}, code
        assert found <= keys["en"], code
    # Every name has its five translations.
    for row in re.findall(r'^  "[^"]+": \[(.*)\],$', script, re.MULTILINE):
        assert len(json.loads(f"[{row}]")) == 5


def test_the_page_carries_its_languages() -> None:
    points = np.zeros((1, 3))
    page = render_viewer(office_scene(), points, np.zeros((1, 3), dtype=np.uint8), "office")
    assert "__PIVOT_LANGUAGES__" not in page and "window.PivotI18n" in page
    assert 'id="languageMenu"' in page
    # The first-visit tour, and the button that shows it again.
    assert 'id="tour"' in page and 'id="tourButton"' in page


def test_the_agent_is_told_the_language_of_the_page_once() -> None:
    sent: list[list[dict[str, Any]]] = []

    def create(**kwargs: Any) -> SimpleNamespace:
        sent.append(list(kwargs["messages"]))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="D'accord.",
                                                             tool_calls=None))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = MetaConfig(api_key="k")
    agent = Agent(Session(office_scene()), config, client)  # type: ignore[arg-type]
    agent.ask("Bonjour", "fr")
    agent.ask("Et la chaise ?", "fr")
    notes = [m["content"] for m in sent[-1] if m["role"] == "system"][1:]
    assert notes == ["The user's page is in French. Answer in French unless the user writes in "
                     "another language, then in theirs; name objects and rooms in that language."]
