"""Regression tests independent of captured room labels or object ids."""

import dataclasses
import json
import numpy as np
import pytest

from pivot.geometry.floor import FloorMap
from pivot.planning.access import check_access
from pivot.planning.placement import check_placement
from pivot.scene.examples import box, sofa_scene
from pivot.scene.graph import SceneObject
from pivot.scene.rooms import find_rooms
from pivot.agent.tools import Session, call_tool
from pivot.planning.way import plan_way
from test_way import room, chair


def test_blocked_door_does_not_seed_access_inside_an_isolated_room():
    blocker = SceneObject(99, "cabinet", box((0.7, 1.5, 1.), (1.1, 3., 2.)), movable=False)
    report = check_access(room(blocker))
    assert report is not None
    assert report.blocked_doors
    assert not report.reachable.any()


def test_wall_evidence_survives_serialization():
    floor = sofa_scene(.9).floor
    assert floor is not None
    walls = np.zeros(floor.shape, dtype=bool)
    walls[1, 1] = True
    restored = FloorMap.from_dict(dataclasses.replace(floor, walls=walls).to_dict())
    assert np.array_equal(restored.walls, walls)


def test_floor_neck_without_walls_or_door_is_not_a_room_boundary():
    scene = sofa_scene(.9)
    assert scene.floor is not None
    evidence = np.zeros(scene.floor.shape, dtype=bool)
    evidence[[0, -1], :] = True
    evidence[:, [0, -1]] = True
    scene = dataclasses.replace(scene, floor=dataclasses.replace(scene.floor, walls=evidence))
    rooms = find_rooms(scene)
    assert rooms is not None
    assert len(rooms.rooms) == 1


def test_real_partition_with_wall_evidence_stays_two_rooms():
    scene = sofa_scene(.9)
    assert scene.floor is not None
    scene = dataclasses.replace(scene, floor=dataclasses.replace(
        scene.floor, walls=~scene.floor.floor))
    rooms = find_rooms(scene)
    assert rooms is not None and len(rooms.rooms) == 2


def test_explicit_door_leaf_is_a_physical_obstacle():
    leaf = SceneObject(90, "door leaf", box((2., 1.5, 1.), (1., .06, 2.)), movable=False)
    mover = SceneObject(91, "table", box((3., 1.5, .4), (.5, .5, .8)))
    scene = room(leaf, mover)
    assert not check_placement(scene, 91, mover.box.moved(np.array([2., 1.5]))).feasible


def test_rejected_door_obstruction_does_not_change_undo_history():
    scene = room(chair(77, 3., 1.5))
    session = Session(scene)
    with pytest.raises(ValueError, match="door"):
        session.record(scene.moved(77, scene[77].box.moved(np.array([.4, 1.5]))), [])
    assert session.scene is scene
    assert not session.history and not session.motions and not session.motion_counts


def test_wheelchair_route_also_clears_a_preexisting_door_obstruction():
    scene = room(chair(77, .4, 1.5))
    way = plan_way(scene, 2)
    assert way.success, way.message
    assert 77 in way.moved


def test_failed_target_proposes_an_accessible_table_without_moving_it():
    barrier = SceneObject(89, "cabinet", box((3., 1.5, 1.), (.2, 3., 2.)), movable=False)
    alternative = SceneObject(57, "table", box((1.7, 1.5, .375), (.8, .8, .75)))
    session = Session(room(barrier, alternative))
    original = session.scene
    result = json.loads(call_tool(session, "make_way", '{"object_id":2}'))
    assert not result["success"]
    assert result["alternatives"][0]["id"] == 57
    assert result["alternatives"][0]["already_accessible"]
    assert result["alternatives"][0]["path_xy"]
    assert session.scene is original and not session.history
