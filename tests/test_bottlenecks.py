"""Failure explanations must name measured gaps without claiming global impossibility."""

import numpy as np

from pivot.agent.tools import Session, call_tool
from pivot.planning.bottlenecks import route_gaps
from pivot.scene.examples import box, rectangle_floor
from pivot.scene.graph import Scene, SceneObject
import json


def narrow_route() -> Scene:
    return Scene.from_objects([
        SceneObject(17, "cabinet", box((3., .6, .5), (.4, 1.2, 1.)), movable=False),
        SceneObject(83, "shelf", box((3., 2.4, .5), (.4, 1.2, 1.)), movable=False),
        SceneObject(91, "table", box((1., 1.5, .4), (.8, .8, .8))),
    ], rectangle_floor(6., 3.))


def test_candidate_gap_is_measured_between_opposing_objects():
    gaps = route_gaps(narrow_route(), np.array([[3., 1.5]]), .8, 91)
    assert len(gaps) == 1
    assert gaps[0]["gap_m"] == .6
    assert {obj["id"] for obj in gaps[0]["objects"]} == {17, 83}
    assert "not ruled out" in gaps[0]["scope"]


def test_objects_on_the_same_side_are_not_a_bottleneck_pair():
    assert not route_gaps(narrow_route(), np.array([[3., 4.]]), .8, 91)


def test_failed_object_motion_exposes_the_measured_candidate_gap():
    session = Session(narrow_route())
    result = json.loads(call_tool(session, "move_object",
                                  '{"object_id":91,"x":5,"y":1.5}'))
    assert not result["moved"]
    assert result["candidate_bottlenecks"][0]["gap_m"] == .6
    assert not session.motions
