"""Accessibility uses supported occupied volumes, retaining conservative margins."""

import numpy as np

from pivot.geometry.shape import VoxelShape
from pivot.planning.access import obstacle_mask
from pivot.planning.way import _in_the_way
from pivot.scene.examples import box, rectangle_floor
from pivot.scene.graph import SceneObject


def test_carved_shape_does_not_occupy_the_whole_bounding_box():
    floor = rectangle_floor(5., 5.)
    bounds = box((2.5, 2.5, .5), (2., 2., 1.))
    points = np.array([[2.5, 2.5, .1], [2.5, 2.5, .5], [2.5, 2.5, .9]])
    shape = VoxelShape.from_world_points(bounds, points, .04)
    obj = SceneObject(1, "chair", bounds, shape=shape)
    occupied = obstacle_mask(floor, obj)
    assert occupied.any()
    assert occupied.sum() < floor.rasterize(bounds.footprint()).sum() / 4


def test_without_a_shape_the_full_box_remains_an_obstacle():
    floor = rectangle_floor(5., 5.)
    obj = SceneObject(1, "chair", box((2.5, 2.5, .5), (1., 1., 1.)))
    assert np.array_equal(obstacle_mask(floor, obj), floor.rasterize(obj.box.footprint()))


def test_empty_parts_of_a_carved_box_do_not_trigger_unnecessary_moves():
    floor = rectangle_floor(5., 5.)
    bounds = box((2.5, 2.5, .5), (2., 2., 1.))
    shape = VoxelShape.from_world_points(bounds, np.array([[2.5, 2.5, .5]]), .04)
    obj = SceneObject(1, "chair", bounds, shape=shape)
    corridor = [box((1.7, 2.5, .5), (.3, .8, 1.))]
    assert _in_the_way([obj], corridor) == [1]
    assert not _in_the_way([obj], corridor, floor)
    assert _in_the_way([obj], [box((2.5, 2.5, .5), (.3, .8, 1.))], floor) == [1]
