"""A group may return towards its original place while preserving the reserved aisle."""

import numpy as np

from pivot.geometry.collision import box_penetration
from pivot.planning.motion import MotionOptions
from pivot.planning.rearrange import restore_group
from pivot.scene.examples import box,rectangle_floor
from pivot.scene.graph import Scene,SceneObject


def test_partial_return_keeps_the_group_and_reserved_passage():
    table = SceneObject(10,'table',box((2.,2.,.375),(.8,1.4,.75)))
    chair = SceneObject(20,'chair',box((2.65,2.,.45),(.5,.5,.9)))
    initial = Scene.from_objects([table,chair],floor=rectangle_floor(8.,5.))
    shifted = initial.moved(20,chair.box.moved(chair.box.center[:2]+[1.2,0.]))
    shifted = shifted.moved(10,table.box.moved(table.box.center[:2]+[1.2,0.]))
    corridor = [box((2.,2.,.5),(.5,3.,1.))]
    result = restore_group(shifted,initial,10,corridor,MotionOptions())
    assert result and {p.object_id for p in result}=={10,20}
    shifts = [p.waypoints[-1].center[:2]-shifted[p.object_id].box.center[:2] for p in result]
    assert np.allclose(shifts[0],shifts[1])
    assert 0<np.linalg.norm(shifts[0])<1.2
    for plan in result:
        assert box_penetration(plan.waypoints[-1],corridor[0])==0.


def test_partial_return_does_not_move_an_independently_parked_chair():
    table = SceneObject(10,'table',box((2.,2.,.375),(.8,1.4,.75)))
    chair = SceneObject(20,'chair',box((2.65,2.,.45),(.5,.5,.9)))
    initial = Scene.from_objects([table,chair],floor=rectangle_floor(8.,5.))
    shifted = initial.moved(10,table.box.moved(np.array([3.2,2.])))
    shifted = shifted.moved(20,chair.box.moved(np.array([5.,4.])))
    assert restore_group(shifted,initial,10,[],MotionOptions()) is None
