"""Observed door panels survive wall snapping as separate collision obstacles."""

import dataclasses
import math
import numpy as np

from pivot.lifting.openings import _door_leaves
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject, is_opening


def test_open_panel_is_not_replaced_by_the_wall_opening():
    panel = SceneObject(37, "door", box((1., .5, 1.), (.9, .06, 2.), math.pi / 2),
                        movable=False)
    opening = dataclasses.replace(panel, box=box((1., .05, 1.), (.9, .08, 2.)))
    leaves = _door_leaves([panel], [opening])
    assert len(leaves) == 1
    leaf = leaves[0]
    assert leaf.box is panel.box
    assert leaf.object_id != opening.object_id
    assert not leaf.movable and not is_opening(leaf)


def test_ambiguous_depth_is_not_an_invented_panel():
    panel = SceneObject(37, "door", box((1., .5, 1.), (.9, .5, 2.), math.pi / 2),
                        movable=False)
    opening = dataclasses.replace(panel, box=box((1., .05, 1.), (.9, .08, 2.)))
    assert not _door_leaves([panel], [opening])


def test_observed_panel_survives_even_if_no_wall_opening_was_kept():
    panel = SceneObject(37, "door", box((1., .5, 1.), (.9, .06, 2.), math.pi / 2),
                        movable=False)
    along, height = np.meshgrid(np.arange(-.4, .41, .05), np.arange(-.9, .91, .05))
    cloud = np.column_stack([np.ones(along.size), .5 + along.ravel(), 1. + height.ravel()])
    leaves = _door_leaves([panel], [], {37: cloud})
    assert len(leaves) == 1 and leaves[0].box is panel.box


def test_an_empty_frame_is_not_filled_in_as_a_closed_panel():
    panel = SceneObject(37, "door", box((1., .5, 1.), (.9, .06, 2.), math.pi / 2),
                        movable=False)
    along, height = np.meshgrid(np.array([-.4, .4]), np.arange(-.9, .91, .05))
    cloud = np.column_stack([np.ones(along.size), .5 + along.ravel(), 1. + height.ravel()])
    assert not _door_leaves([panel], [], {37: cloud})
