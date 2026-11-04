"""Access proposals are advisory; verified routes describe the actual final layout."""

import json
from dataclasses import replace

import numpy as np
import pytest
from test_agent import ScriptedModel, message, tool_call
from test_way import DOOR, chair, room
from test_lift import office as divided_office

from pivot.agent.loop import Agent
from pivot.agent.tools import Session, call_tool, make_way, suggest_access
from pivot.geometry.shape import VoxelShape
from pivot.meta.client import MetaConfig
from pivot.planning.way import clear_way, plan_way
from pivot.planning.motion import MotionOptions
from pivot.planning.rearrange import park_aside
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject
from pivot.viz.plan import instructions_page


def test_preview_has_a_route_but_does_not_apply_moves_or_publish_it():
    row = [chair(3+i, 3., y) for i, y in enumerate((.3, 1.1, 1.9, 2.7))]
    session = Session(room(*row))
    original = session.scene
    result = make_way(session, 2, execute=False)
    assert result["success"] and result["moved_aside"]
    assert not result["moved"] and not result["executed"]
    assert result["steps"] and all(step["object_id"] != 2 for step in result["steps"])
    assert result["path_xy"][0] == DOOR.box.center[:2].tolist()
    assert session.scene is original and not session.history and not session.access_routes()


def test_confirmed_route_is_visible_in_printable_plan_even_without_moves():
    session = Session(room())
    make_way(session, 2)
    routes = session.access_routes()
    assert len(routes) == 1 and routes[0]["object_id"] == 2
    assert np.allclose(routes[0]["path_xy"][-1], routes[0]["spot"]["center_xy"], atol=.001)
    page = instructions_page(session.scene, [], access_routes=routes)
    assert 'class="access-route"' in page
    assert '#2 · 90 cm' in page


def test_later_manual_edit_cannot_reblock_a_confirmed_route():
    session = Session(room(chair(3, 3., 1.5)))
    make_way(session, 2)
    original = session.scene
    result = json.loads(call_tool(session, "move_object", '{"object_id":3,"x":4.6,"y":1.5}'))
    assert "error" in result
    assert session.scene is original and not session.history
    assert session.access_routes()


def test_cancelled_rearrangement_does_not_publish_a_provisional_route():
    row = [chair(3+i, 3., y) for i, y in enumerate((.3, 1.1, 1.9, 2.7))]
    session = Session(room(*row))
    original = session.scene
    model = ScriptedModel([message(calls=[tool_call("a", "make_way", object_id=2)]),
                           message("Finished")])
    run = Agent(session, MetaConfig(api_key="test"), client=model,
                max_steps=1).ask("Make room")
    assert not run.finished and session.scene is original
    assert not session.access_goals and not session.access_routes()


def test_recommendation_checks_clear_places_without_moving_furniture():
    alternative = SceneObject(57, "table", box((1.7, 1.5, .375), (.8,.8,.75)))
    session = Session(room(alternative))
    original = session.scene
    result = suggest_access(session)
    assert result["alternatives"] and not result["executed"]
    assert result["alternatives"][0]["already_accessible"]
    assert all(item["moves_required"] == 0 for item in result["alternatives"])
    assert session.scene is original and not session.history and not session.access_goals


def test_recommendation_can_propose_a_rearrangement_without_executing_it():
    session = Session(room(*[chair(3+i,3.,y) for i,y in enumerate((.3,1.1,1.9,2.7))]))
    original = session.scene
    proposal = suggest_access(session)
    assert proposal["alternatives"]
    candidate = proposal["alternatives"][0]
    assert candidate["moves_required"] == 1 and candidate["requires_confirmation"]
    assert not candidate["already_accessible"] and candidate["path_xy"]
    assert session.scene is original and not session.history


def test_failed_target_keeps_the_answer_proposing_a_verified_alternative():
    barrier = SceneObject(89,"cabinet",box((3.,1.5,1.),(.2,3.,2.)),movable=False)
    alternative = SceneObject(57,"table",box((1.7,1.5,.375),(.8,.8,.75)))
    session = Session(room(barrier,alternative))
    original = session.scene
    text = "La table 57 a un accès vérifié sans déplacement. Souhaitez-vous cette alternative ?"
    model = ScriptedModel([message(calls=[tool_call("a","make_way",object_id=2)]),message(text)])
    run = Agent(session,MetaConfig(api_key="test"),client=model).ask("Reach table 2")
    assert run.answer == text and run.finished
    assert run.steps[0].result["alternatives"][0]["id"] == 57
    assert session.scene is original and not session.motions


def test_facing_reference_changes_which_places_can_be_confirmed():
    west = SceneObject(81, "painting", box((.05,1.5,1.8), (.01,.1,.1)), movable=False)
    east = SceneObject(82, "painting", box((5.98,1.5,1.8), (.01,.1,.1)), movable=False)
    scene = room(west, east)
    assert clear_way(scene, 2, facing_object_id=82).success
    assert not clear_way(scene, 2, facing_object_id=81).success


def test_too_narrow_door_does_not_prove_wheelchair_access():
    scene = room()
    door = replace(DOOR, box=box((.05,1.5,1.), (.7,.1,2.), yaw=np.pi/2))
    scene = replace(scene, objects={**scene.objects, 1:door})
    assert not plan_way(scene, 2).success


def test_voxel_margin_does_not_reject_all_places_at_their_own_table():
    scene = room()
    table = scene[2]
    x = np.linspace(-.4,.4,15)
    y = np.linspace(-.8,.8,25)
    xy = np.array(np.meshgrid(x,y)).reshape(2,-1).T
    points = np.column_stack([xy+table.box.center[:2], np.full(len(xy),.75)])
    shape = VoxelShape.from_world_points(table.box,points,.04)
    scene = replace(scene,objects={**scene.objects,2:replace(table,shape=shape)})
    assert clear_way(scene,2).success


def test_initial_layout_comparison_does_not_restore_or_move_anything():
    session = Session(room(chair(3,3.,1.5)))
    make_way(session,2)
    result = json.loads(call_tool(session,"move_object",'{"object_id":3,"x":2.0,"y":2.5}'))
    assert result["moved"]
    current = session.scene
    proposal = suggest_access(session,from_initial=True)
    assert proposal["source_layout"] == "initial"
    assert proposal["requires_restore_confirmation"]
    assert not proposal["executed"] and session.scene is current


def test_light_blocker_can_be_parked_across_a_gap_that_requires_carrying():
    scene = divided_office()
    # Left-hand floor is reserved for the person; only the right side can park it.
    sweep = [box((1.,1.5,.5),(2.4,3.,1.))]
    plans = park_aside(scene,5,sweep,MotionOptions(max_expansions=20000))
    assert plans
    assert plans[-1].waypoints[-1].center[0] > 2.4
    assert any(pose.z_min > .7 for pose in plans[-1].waypoints)


def test_blocker_can_depart_from_under_an_obstacle_before_being_lifted():
    scene = divided_office()
    shelf = SceneObject(9,"shelf",box((.6,2.4,1.),(.8,.8,.2)),movable=False)
    scene = replace(scene,objects={**scene.objects,9:shelf})
    plans = park_aside(scene,5,[box((1.,1.5,.5),(2.4,3.,1.))],MotionOptions())
    assert plans
    poses = plans[-1].waypoints
    assert poses[-1].center[0] > 2.4
    lifted = next(i for i,pose in enumerate(poses) if pose.z_min > .7)
    assert lifted > 1
    assert np.linalg.norm(poses[lifted-1].center[:2]-poses[0].center[:2]) > .3


@pytest.mark.parametrize("width", [0., -.1])
def test_invalid_dimensions_do_not_modify_the_scene(width):
    session = Session(room())
    original = session.scene
    with pytest.raises(ValueError):
        make_way(session, 2, width_m=width)
    assert session.scene is original and not session.history
