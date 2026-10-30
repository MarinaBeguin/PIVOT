"""Tests of the HTTP server of the interactive viewer."""

import numpy as np
from fastapi.testclient import TestClient

from pivot.agent.loop import Agent
from pivot.geometry.types import FloatArray
from pivot.meta.client import MetaConfig
from pivot.scene.examples import office_scene, sofa_scene
from pivot.server.app import MAX_AUDIO_BYTES, Room, create_app
from pivot.server.limits import Budget, Meter, SpendLedger
from pivot.viz.pointcloud import ColorArray
from test_agent import ScriptedModel, message, tool_call


def office(points: FloatArray, colors: ColorArray) -> list[Room]:
    return [Room("office", "Office", office_scene(), points, colors)]


def client() -> TestClient:
    points = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    return TestClient(create_app(office(points, colors)))


def test_page_is_the_interactive_viewer() -> None:
    page = client().get("/").text
    assert '"interactive": true' in page
    assert "__PIVOT_SCENE_DATA__" not in page


def test_move_returns_updated_objects_and_can_be_undone() -> None:
    app = client()
    moved = app.post("/api/tools/move_object", json={"object_id": 3, "x": 3.0, "y": 1.0}).json()
    assert moved["moved"]
    chair = next(o for o in moved["objects"] if o["id"] == 3)
    assert chair["box"]["center"][:2] == [3.0, 1.0]
    undone = app.post("/api/tools/undo", json={}).json()
    assert next(o for o in undone["objects"] if o["id"] == 3)["box"]["center"][:2] == [1.5, 1.7]


def test_checks_do_not_move_and_errors_are_reported() -> None:
    app = client()
    plan = app.post("/api/tools/plan_motion", json={"object_id": 3, "x": 3.0, "y": 1.0}).json()
    assert plan["status"] == "success" and "objects" not in plan
    assert app.post("/api/tools/teleport", json={}).status_code == 404
    assert "error" in app.post("/api/tools/check_placement", json={"object_id": 99, "x": 0,
                                                                     "y": 0}).json()


def test_reset_restores_the_initial_scene() -> None:
    app = client()
    app.post("/api/tools/move_object", json={"object_id": 3, "x": 3.0, "y": 1.0})
    objects = app.post("/api/reset").json()
    assert next(o for o in objects if o["id"] == 3)["box"]["center"][:2] == [1.5, 1.7]


def test_requests_in_words_are_off_unless_an_agent_is_given() -> None:
    app = client()
    assert '"chat": false' in app.get("/").text
    assert app.post("/api/chat", json={"message": "Hi"}).status_code == 404


def test_chat_answers_and_returns_the_moves_to_play_back() -> None:
    model = ScriptedModel([
        message(calls=[tool_call("a", "move_object", object_id=3, x=3.0, y=1.0)]),
        message("Done: the chair is at (3.0, 1.0)."),
    ])
    points = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    app = TestClient(create_app(office(points, colors), agent_factory=lambda session: Agent(
        session, MetaConfig(api_key="test"), client=model)))  # type: ignore[arg-type]
    assert '"chat": true' in app.get("/").text
    answer = app.post("/api/chat", json={"message": "Move the chair to (3, 1)."}).json()
    assert answer["answer"].startswith("Done")
    assert [s["tool"] for s in answer["steps"]] == ["move_object"]
    assert [m["object_id"] for m in answer["motions"]] == [3]
    chair = next(o for o in answer["objects"] if o["id"] == 3)
    assert chair["box"]["center"][:2] == [3.0, 1.0]
    # The buttons work on the same scene: undo reverts the agent's move.
    undone = app.post("/api/tools/undo", json={}).json()
    assert next(o for o in undone["objects"] if o["id"] == 3)["box"]["center"][:2] == [1.5, 1.7]


def test_spoken_requests_are_transcribed_with_the_scene_labels() -> None:
    heard: list[tuple[bytes, list[str]]] = []

    def transcriber(audio: bytes, keywords: list[str]) -> str:
        heard.append((audio, keywords))
        return "move the chair"

    points = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    app = TestClient(create_app(office(points, colors), agent_factory=lambda session: Agent(
        session, MetaConfig(api_key="test"), client=ScriptedModel([])),  # type: ignore[arg-type]
        transcriber=transcriber))
    assert '"voice": true' in app.get("/").text
    assert app.post("/api/transcribe", content=b"RIFF").json() == {"text": "move the chair"}
    assert heard[0][0] == b"RIFF" and "office chair" in heard[0][1]
    assert client().post("/api/transcribe", content=b"RIFF").status_code == 404


def test_moving_plan_lists_the_moves_made() -> None:
    app = client()
    assert "No move yet" in app.get("/instructions").text
    app.post("/api/tools/move_object", json={"object_id": 3, "x": 3.0, "y": 1.0})
    page = app.get("/instructions").text
    assert "<ol><li>Slide the office chair (#3)" in page


def test_each_visitor_rearranges_their_own_room() -> None:
    points = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    app = create_app(office(points, colors))
    alice, bob = TestClient(app), TestClient(app)
    alice.post("/api/tools/move_object", json={"object_id": 3, "x": 3.0, "y": 1.0})
    chair = next(o for o in alice.get("/api/objects").json() if o["id"] == 3)
    assert chair["box"]["center"][:2] == [3.0, 1.0]
    untouched = next(o for o in bob.get("/api/objects").json() if o["id"] == 3)
    assert untouched["box"]["center"][:2] == [1.5, 1.7]


def test_public_requests_are_charged_and_refused_over_budget() -> None:
    model = ScriptedModel([message("Hello."), message("Again.")])
    ledger = SpendLedger(Budget(total_usd=1.0, visitor_daily_usd=1.0, visitor_daily_requests=1))
    meter = Meter(ledger, "salt", proxy_hops=0, chat_price=lambda usage: 0.25)
    points = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    app = TestClient(create_app(office(points, colors), agent_factory=lambda session: Agent(
        session, MetaConfig(api_key="test"), client=model),  # type: ignore[arg-type]
        meter=meter))
    assert app.get("/api/limits").json() == {"public": True, "requests_left": 1,
                                             "budget_spent": False}
    assert app.post("/api/chat", json={"message": "Hi"}).json()["answer"] == "Hello."
    assert ledger.total == 0.25
    assert app.get("/api/limits").json()["requests_left"] == 0
    refused = app.post("/api/chat", json={"message": "Hi again"})
    assert refused.status_code == 429 and "requests today" in refused.json()["error"]
    assert ledger.total == 0.25


def test_long_recordings_are_refused() -> None:
    points = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    app = TestClient(create_app(office(points, colors), agent_factory=lambda session: Agent(
        session, MetaConfig(api_key="test"), client=ScriptedModel([])),  # type: ignore[arg-type]
        transcriber=lambda audio, keywords: "never"))
    assert app.post("/api/transcribe", content=b"0" * (MAX_AUDIO_BYTES + 1)).status_code == 413
    assert app.get("/api/limits").json() == {"public": False}



def test_each_room_has_its_page_and_its_own_session() -> None:
    points = np.zeros((3, 3), dtype=np.float32)
    colors = np.zeros((3, 3), dtype=np.uint8)
    app = TestClient(create_app([*office(points, colors),
                                 Room("two-rooms", "Two rooms", sofa_scene(1.0), points, colors)]))
    page = app.get("/?scene=two-rooms").text
    assert '"scene": "two-rooms"' in page and '"title": "Office"' in page
    assert '"scene": "office"' in app.get("/").text  # The first room by default.
    app.post("/api/tools/move_object?scene=office", json={"object_id": 3, "x": 3.0, "y": 1.0})
    sofa = app.get("/api/objects?scene=two-rooms").json()
    assert [o["label"] for o in sofa] == ["sofa"]
    chair = next(o for o in app.get("/api/objects").json() if o["id"] == 3)
    assert chair["box"]["center"][:2] == [3.0, 1.0]  # The office kept its move.
    app.post("/api/reset?scene=two-rooms")
    chair = next(o for o in app.get("/api/objects?scene=office").json() if o["id"] == 3)
    assert chair["box"]["center"][:2] == [3.0, 1.0]  # Resetting a room leaves the others.
