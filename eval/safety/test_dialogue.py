"""Deterministic multi-turn contract; does not measure a live model's language quality."""

import pytest

from pivot.agent.loop import Agent
from pivot.agent.tools import Session
from pivot.meta.client import MetaConfig
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject
from pivot.viz.plan import instructions_page
from test_agent import ScriptedModel, message, tool_call
from test_way import chair, room


@pytest.mark.parametrize("with_move", [True, pytest.param(False, marks=pytest.mark.xfail(
    strict=True, reason="Undo does not clear a route confirmed without furniture edits"))])
def test_failed_target_alternative_confirmation_and_undo(with_move):
    barrier = SceneObject(89, 'cabinet', box((3., 1.5, 1.), (.2, 3., 2.)), movable=False)
    alternative = SceneObject(57, 'table', box((1.7, 1.5, .375), (.8, .8, .75)))
    session = Session(room(barrier, alternative, chair(78, 4.5, .5)))
    initial = session.scene
    model = ScriptedModel([
        message(calls=[tool_call('a', 'make_way', object_id=2)]),
        message('La table 57 est une alternative accessible. Confirmer ?'),
        message(calls=([tool_call('b0', 'move_object', object_id=78, x=4.5, y=2.5)]
                       if with_move else []) + [tool_call('b', 'make_way', object_id=57)]),
        message('La place et le trajet sont vérifiés. Terminé.'),
        message(calls=[tool_call('c', 'undo')]),
        message('Annulé.'),
    ])
    agent = Agent(session, MetaConfig(api_key='test'), client=model)
    proposal = agent.ask('Installe un fauteuil à la table 2', language='fr')
    assert proposal.finished
    assert proposal.steps[0].result['alternatives'][0]['id'] == 57
    assert session.scene is initial and not session.access_routes()
    confirmed = agent.ask('Oui, utilise cette alternative', language='fr')
    assert confirmed.finished and confirmed.steps[-1].result['success']
    routes = session.access_routes()
    assert len(routes) == 1 and routes[0]['object_id'] == 57
    assert 'class="access-route"' in instructions_page(session.scene, [], access_routes=routes)
    undone = agent.ask('Annule', language='fr')
    assert undone.finished and undone.steps[0].result['undone']
    assert session.scene is initial and not session.access_routes() and not session.motions
