"""Tests for the agent's tools and loop, with a scripted stand-in for the model."""

import json
from types import SimpleNamespace
from typing import Any

import pytest

from pivot.agent.loop import Agent
from pivot.agent.tools import TOOLS, Session, call_tool
from pivot.meta.client import MetaConfig
from pivot.scene.examples import office_scene, sofa_scene


def call(session: Session, name: str, **arguments: Any) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
    return result


def test_every_tool_has_a_valid_schema() -> None:
    names = [tool.name for tool in TOOLS]
    assert len(names) == len(set(names))
    for tool in TOOLS:
        schema = tool.schema()["function"]
        assert schema["parameters"]["type"] == "object"
        assert set(schema["parameters"]["required"]) <= set(schema["parameters"]["properties"])


def test_list_objects_reports_support_and_floor() -> None:
    result = call(Session(office_scene()), "list_objects")
    lamp = next(o for o in result["objects"] if o["label"] == "lamp")
    desk = next(o for o in result["objects"] if o["label"] == "desk")
    assert lamp["resting_on"] == desk["id"]
    assert desk["carries"] == [lamp["id"]]
    assert result["floor_bounds_m"]["max_xy"] == pytest.approx([5.0, 4.0], abs=0.06)


def test_check_placement_quantifies_conflicts() -> None:
    result = call(Session(office_scene()), "check_placement", object_id=3, x=1.5, y=2.3)
    assert not result["feasible"]
    assert result["conflicts"][0]["other_label"] == "desk"
    assert result["conflicts"][0]["penetration_m"] > 0.3


def test_move_object_requires_a_valid_motion() -> None:
    session = Session(sofa_scene(0.8))
    result = call(session, "move_object", object_id=1, x=6.0, y=2.0)
    assert not result["moved"]
    assert result["plan"]["status"] == "no_path"
    assert not session.history


def test_move_then_undo() -> None:
    session = Session(office_scene())
    result = call(session, "move_object", object_id=3, x=3.0, y=1.0)
    assert result["moved"], result
    assert session.scene[3].box.center[:2] == pytest.approx([3.0, 1.0])
    assert len(session.motions) == 1
    assert call(session, "undo")["undone"]
    assert session.scene[3].box.center[:2] == pytest.approx([1.5, 1.7])
    assert not call(session, "undo")["undone"]


def test_moving_desk_reports_carried_lamp() -> None:
    session = Session(office_scene())
    result = call(session, "move_object", object_id=1, x=2.5, y=1.5)
    assert result["moved"], result
    assert result["carried_along"] == [2]


def test_errors_are_returned_to_the_model() -> None:
    session = Session(office_scene())
    assert "error" in call(session, "check_placement", object_id=99, x=0.0, y=0.0)
    assert "error" in json.loads(call_tool(session, "teleport", "{}"))
    assert "error" in json.loads(call_tool(session, "check_placement", "{not json"))


class ScriptedModel:
    """Stand-in for the Chat Completions API that replays scripted messages."""

    def __init__(self, script: list[SimpleNamespace]):
        self.script = script
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=self.script.pop(0))])


def tool_call(call_id: str, name: str, **arguments: Any) -> SimpleNamespace:
    function = SimpleNamespace(name=name, arguments=json.dumps(arguments))
    payload = {"id": call_id, "type": "function",
               "function": {"name": name, "arguments": function.arguments}}
    return SimpleNamespace(id=call_id, function=function, model_dump=lambda: payload)


def message(content: str = "", calls: list[SimpleNamespace] | None = None) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=calls)


def test_agent_loop_runs_tools_until_answer() -> None:
    model = ScriptedModel([
        message(calls=[tool_call("a", "list_objects")]),
        message(calls=[tool_call("b", "plan_motion", object_id=3, x=2.8, y=3.5)]),
        message("Oui : la chaise peut aller sous la fenêtre."),
    ])
    agent = Agent(Session(office_scene()), MetaConfig(api_key="test", agent_model="muse-test"),
                  client=model)  # type: ignore[arg-type]
    run = agent.ask("Peut-on mettre la chaise près de la fenêtre ?")
    assert run.finished
    assert [s.tool for s in run.steps] == ["list_objects", "plan_motion"]
    assert run.steps[1].result["status"] == "success"
    assert run.answer.startswith("Oui")
    # Each tool call is answered by a tool message carrying its id.
    roles = [(m["role"], m.get("tool_call_id")) for m in agent.messages[2:]]
    assert roles == [("assistant", None), ("tool", "a"), ("assistant", None), ("tool", "b"),
                     ("assistant", None)]
    assert model.requests[0]["model"] == "muse-test"
    assert {t["function"]["name"] for t in model.requests[0]["tools"]} >= {"plan_motion"}


def test_agent_out_of_steps_still_summarizes_what_it_did() -> None:
    model = ScriptedModel([message(calls=[tool_call(str(i), "list_objects")]) for i in range(3)]
                          + [message("J'ai listé les objets ; il reste à déplacer la chaise.")])
    config = MetaConfig(api_key="test")
    agent = Agent(Session(office_scene()), config, client=model,  # type: ignore[arg-type]
                  max_steps=3)
    run = agent.ask("Range la pièce.")
    assert not run.finished
    assert len(run.steps) == 3
    assert run.answer.startswith("J'ai listé")
    # The summary is asked for without tools.
    assert model.requests[-1]["tool_choice"] == "none"


def test_target_too_close_to_a_wall_snaps_to_the_nearest_feasible_pose() -> None:
    session = Session(office_scene())
    # The chair is 60 cm wide: centered 10 cm from the wall, it would cross it.
    strict = call(session, "plan_motion", object_id=3, x=0.1, y=1.0)
    assert strict["status"] == "invalid_goal"
    snapped = call(session, "plan_motion", object_id=3, x=0.1, y=1.0, yaw_deg=0, snap_m=0.5)
    assert snapped["status"] == "success"
    adjusted = snapped["goal_adjusted"]
    # Half the chair plus the 2.5 cm floor margin, at most 15 cm more.
    assert 0.27 <= adjusted["x"] <= 0.45 and adjusted["yaw_deg"] == 0.0
    assert adjusted["shift_m"] <= 0.36


def test_snapping_gives_up_beyond_its_radius() -> None:
    session = Session(office_scene())
    # The middle of the desk is more than 5 cm from any free spot.
    result = call(session, "move_object", object_id=3, x=1.5, y=2.5, snap_m=0.05)
    assert not result["moved"] and "goal_adjusted" not in result
