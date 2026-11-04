"""Saved exports and reconstruction cleanup must preserve supported instances."""

from dataclasses import replace

import numpy as np

from pivot.lifting.openings import Walls, clean_openings
from pivot.scene.examples import box, rectangle_floor, sofa_scene
from pivot.scene.graph import SceneObject
from pivot.scene.rooms import find_rooms
from pivot.viz.web import scene_payload


def test_valid_door_survives_a_larger_wall_misclassified_as_a_door():
    rng = np.random.default_rng(4)
    wall = rng.uniform([.5,2.95,0.],[2.5,3.05,3.3],(1600,3))
    door = rng.uniform([1.5,2.95,0.],[2.3,3.05,2.1],(500,3))
    objects = [SceneObject(10,"door",box((1.5,3.,1.65),(2.,.1,3.3)),movable=False),
               SceneObject(20,"door",box((1.9,3.,1.05),(.8,.1,2.1)),movable=False)]
    walls = Walls(0.,np.array([0.,0.]),np.array([4.,3.]),rectangle_floor(4.,3.))
    result = clean_openings(objects,{10:wall,20:door},walls)
    openings = [o for o in result if o.label=='door']
    assert len(openings)==1 and openings[0].object_id==20
    assert openings[0].box.size[2] < 2.2


def outer_wall_cloud(scene):
    floor = scene.floor
    border = np.zeros(floor.shape,dtype=bool)
    border[[0,-1],:] = True
    border[:,[0,-1]] = True
    rows,cols = np.nonzero(border)
    xy = floor.origin+(np.column_stack([cols,rows])+.5)*floor.resolution
    return np.repeat(np.column_stack([xy,np.full(len(xy),1.8)]),30,axis=0)


def test_legacy_export_uses_the_same_wall_evidence_as_the_server():
    scene = sofa_scene(.9)
    assert len(find_rooms(scene).rooms)==2
    points = outer_wall_cloud(scene)
    colors = np.zeros((len(points),3),dtype=np.uint8)
    payload = scene_payload(scene,points,colors,"Room")
    assert payload['rooms']==[]
    assert len(find_rooms(scene.with_wall_evidence(points)).rooms)==1
    assert scene.floor.walls is None


def test_stored_partition_evidence_is_not_overwritten_by_viewer_cloud():
    scene = sofa_scene(.9)
    scene = replace(scene,floor=replace(scene.floor,walls=~scene.floor.floor))
    prepared = scene.with_wall_evidence(outer_wall_cloud(scene))
    assert prepared is scene
    assert len(find_rooms(prepared).rooms)==2
