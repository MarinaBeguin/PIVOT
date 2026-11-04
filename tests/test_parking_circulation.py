"""Parking blockers must not consume another furniture approach."""

import math
import numpy as np
from dataclasses import replace
from test_lift import office as divided_office

from pivot.agent.tools import Session, make_way
from pivot.planning.way import clear_way
from pivot.planning.parking_pockets import parking_pockets
from pivot.planning.parking_motion import carry_between
from pivot.planning.cspace import ConfigurationSpace
from pivot.planning.motion import MotionOptions, MotionStatus
from pivot.scene.examples import box, rectangle_floor
from pivot.scene.graph import Scene, SceneObject
from pivot.geometry.collision import box_distance


def test_chair_is_parked_away_from_another_tables_approach():
    door = SceneObject(1,"door",box((.05,2.,1.),(.9,.1,2.),yaw=math.pi/2),movable=False)
    target = SceneObject(2,"table",box((5.5,2.,.375),(.8,1.6,.75)))
    chair = SceneObject(3,"chair",box((.8,2.,.45),(.5,.5,.9)))
    other = SceneObject(8,"table",box((1.,3.5,.375),(1.,.8,.75)))
    scene = Scene.from_objects([door,target,chair,other],floor=rectangle_floor(6.,4.))
    session = Session(scene)
    result = make_way(session,2)
    assert result["success"]
    # Store it beside furniture, with walking room, rather than loose in the aisle.
    assert box_distance(session.scene[3].box,other.box) <= .15
    assert clear_way(session.scene,8,.6,(.6,.6)).success


def test_parking_requires_clearance_even_without_an_accessible_door():
    scene = Scene.from_objects([],floor=rectangle_floor(1.2,4.))
    pockets = parking_pockets(scene,99)
    assert pockets is not None
    assert not pockets.accepts(box((.6,2.,.45),(.5,.5,.9)))
    assert pockets.accepts(box((.25,2.,.45),(.5,.5,.9)))


def test_being_next_to_a_table_is_not_enough_when_it_closes_the_aisle():
    left = SceneObject(1,"table",box((1.,2.,.375),(.8,2.,.75)))
    right = SceneObject(2,"table",box((2.7,2.,.375),(.8,2.,.75)))
    scene = Scene.from_objects([left,right],floor=rectangle_floor(4.,4.))
    pockets = parking_pockets(scene,99)
    assert pockets is not None
    # A chair fits physically in this 90 cm gap, but leaves no pedestrian strip.
    assert not pockets.accepts(box((1.8,2.,.45),(.5,.5,.9)))


def test_carried_object_can_land_beside_an_overhang_then_slide_under_it():
    scene = divided_office()
    shelf = SceneObject(9,"shelf",box((3.4,2.4,1.),(.8,.8,.2)),movable=False)
    scene = replace(scene,objects={**scene.objects,9:shelf})
    goal = scene[5].box.moved(np.array([3.4,2.4]))
    space = ConfigurationSpace.build(scene,5)
    plan = carry_between(scene,5,goal,.85,MotionOptions(),space)
    assert plan is not None and plan.status is MotionStatus.SUCCESS
    assert np.allclose(plan.waypoints[-1].center,goal.center)
    assert any(pose.z_min>.8 for pose in plan.waypoints)
    assert plan.waypoints[-2].z_min == 0.
