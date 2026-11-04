"""Blank model responses must not silently complete a rearrangement."""

import pytest

from pivot.agent.loop import Agent, AgentRun, Step
from pivot.agent.tools import Session, call_tool
from pivot.meta.client import MetaConfig
from pivot.scene.examples import office_scene
from test_agent import ScriptedModel, message, tool_call
from test_way import room, chair


@pytest.mark.parametrize("blank", ["", "  \n "])
def test_blank_final_answer_has_localized_fallback(blank):
    model = ScriptedModel([message(blank), message(blank)])
    agent = Agent(Session(office_scene()), MetaConfig(api_key="test"), client=model)
    run = agent.ask("Déplace la chaise", language="fr")
    assert run.answer.startswith("Je n'ai pas pu confirmer")
    assert not run.finished
    assert model.requests[-1]["tool_choice"] == "none"


def test_budget_exhaustion_and_blank_wrapup_still_answer():
    model = ScriptedModel([message(calls=[tool_call("a", "list_objects")]), message(" \n")])
    agent = Agent(Session(office_scene()), MetaConfig(api_key="test"), client=model,
                  max_steps=1)
    run = agent.ask("Make room", language="en")
    assert run.answer.strip()
    assert not run.finished


def test_exhausted_rearrangement_does_not_apply_partial_moves():
    model = ScriptedModel([message(calls=[tool_call("a", "move_object", object_id=3,
                                                   x=3., y=1.)]), message("Done")])
    session = Session(office_scene())
    original = session.scene
    agent = Agent(session, MetaConfig(api_key="test"), client=model, max_steps=1)
    run = agent.ask("Déplace la chaise", language="fr")
    assert not run.finished
    assert session.scene is original
    assert not session.motions and not session.history
    assert "aucun nouveau" in run.answer
    assert run.steps[0].result["applied"] is False


def test_model_error_discards_plan_and_restores_conversation():
    model = ScriptedModel([message(calls=[tool_call("a", "move_object", object_id=3,
                                                   x=3., y=1.)])])
    session = Session(office_scene())
    original = session.scene
    agent = Agent(session, MetaConfig(api_key="test"), client=model)
    with pytest.raises(IndexError):
        agent.ask("Move it", language="fr")
    assert session.scene is original and not session.history
    assert len(agent.messages) == 1
    model.script.append(message("Ready"))
    assert agent.ask("Hello").answer == "Ready"


def test_completed_multi_move_request_is_one_undo_action():
    model = ScriptedModel([message(calls=[
        tool_call("a", "move_object", object_id=3, x=3., y=1.),
        tool_call("b", "move_object", object_id=4, x=4., y=3.),
    ]), message("Done")])
    session = Session(office_scene())
    original = session.scene
    agent = Agent(session, MetaConfig(api_key="test"), client=model)
    assert agent.ask("Move both").finished
    assert len(session.motions) == 2
    assert len(session.history) == 1
    call_tool(session, "undo", "{}")
    assert session.scene is original
    assert not session.motions


def test_failed_last_edit_does_not_apply_an_earlier_provisional_move():
    model = ScriptedModel([message(calls=[
        tool_call("a", "move_object", object_id=3, x=3., y=1.),
        tool_call("b", "move_object", object_id=999, x=4., y=3.),
    ]), message("The second move failed")])
    session = Session(office_scene())
    original = session.scene
    agent = Agent(session, MetaConfig(api_key="test"), client=model)
    run = agent.ask("Move both")
    assert not run.finished
    assert session.scene is original and not session.motions


def test_prepared_request_does_not_overwrite_a_concurrent_manual_edit():
    model = ScriptedModel([message(calls=[
        tool_call("a", "move_object", object_id=3, x=3., y=1.),
    ]), message("Done")])
    session = Session(office_scene())
    agent = Agent(session, MetaConfig(api_key="test"), client=model)

    def manual_edit(_tool, _arguments):
        call_tool(session, "move_object", '{"object_id": 4, "x": 4, "y": 3}')

    run = agent.ask("Move the chair", on_step=manual_edit)
    assert not run.finished
    assert session.scene[3].box.center[0] == 1.5
    assert session.scene[4].box.center[0] == 4.
    assert len(session.motions) == 1


def test_access_goal_is_rechecked_after_later_provisional_moves():
    model = ScriptedModel([message(calls=[
        tool_call("a", "make_way", object_id=2),
        tool_call("b", "move_object", object_id=3, x=4.6, y=1.5),
    ]), message("The table is accessible")])
    session = Session(room(*[chair(3+i, 3., y) for i, y in enumerate((.3, 1.1, 1.9, 2.7))]))
    original = session.scene
    agent = Agent(session, MetaConfig(api_key="test"), client=model)
    run = agent.ask("Make the table accessible")
    assert not run.finished
    assert session.scene is original and not session.motions


def test_fallback_can_explain_a_measured_local_gap():
    agent = Agent(Session(office_scene()), MetaConfig(api_key="test"), client=ScriptedModel([]))
    run = AgentRun(answer="", steps=[Step("move_object", {}, {
        "moved": False, "candidate_bottlenecks": [{
            "objects": [{"id": 17, "label": "cabinet"}, {"id": 83, "label": "shelf"}],
            "gap_m": .6, "required_width_m": .8,
        }],
    })])
    explanation = agent._fallback(run)  # pylint: disable=protected-access
    assert "cabinet (#17)" in explanation and "shelf (#83)" in explanation
    assert "60 cm available, 80 cm required" in explanation


@pytest.mark.parametrize("language", ["en", "fr", "de", "es", "it", "zh"])
def test_fallback_does_not_report_provisional_moves_as_applied(language):
    session = Session(office_scene())
    original = session.scene
    model = ScriptedModel([message(calls=[
        tool_call("a", "move_object", object_id=3, x=3., y=1.),
        tool_call("b", "undo"),
    ]), message("")])
    agent = Agent(session, MetaConfig(api_key="test"), client=model, max_steps=1)
    run = agent.ask("Try a move, then undo it", language=language)
    assert not run.finished
    assert session.scene is original and not session.motions
    assert "#3" not in run.answer
    assert "recorded moves" not in run.answer
    assert "déplacements enregistrés" not in run.answer
    assert "annuler" not in run.answer
