"""Display geometry must survive brief reflections and pre-refinement noise."""

import numpy as np

from pivot.lifting.agreement import agree_with_camera
from pivot.lifting.objects import contained_tracks
from pivot.lifting.plausible import drop_implausible
from pivot.lifting.scene_builder import settle
from pivot.perception.tracks import Track
from pivot.scene.examples import box
from pivot.scene.graph import SceneObject


def test_brief_reflection_does_not_remove_the_persistent_object():
    panel = np.zeros((40,40),dtype=bool)
    panel[10:30,10:30] = True
    mirror = np.ones_like(panel)
    tracks = [Track(1,'television',{f:panel for f in range(20)}),
              Track(2,'mirror',{f:mirror for f in range(3)})]
    assert 0 not in contained_tracks(tracks)


def test_a_consistently_contained_part_is_still_removed():
    panel = np.zeros((40,40),dtype=bool)
    panel[10:30,10:30] = True
    tracks = [Track(1,'shelf',{f:panel for f in range(3)}),
              Track(2,'lamp',{f:np.ones_like(panel) for f in range(20)})]
    assert contained_tracks(tracks)=={0}


def test_a_display_with_an_unseen_support_does_not_extend_to_the_floor():
    display = SceneObject(1,'television',box((2.,2.,.9),(1.3,.1,.7)))
    result = settle([display],.1)
    assert result[0].box is display.box and result[0].movable


def test_a_borderline_candidate_must_pass_strict_limits_after_refinement():
    rng = np.random.default_rng(9)
    panel = rng.uniform([0.,0.,.5],[1.3,.1,1.2],(4000,3))
    background = rng.uniform([0.,.1,.5],[1.3,.53,1.2],(4000,3))
    candidate = SceneObject(1,'television',box((.65,.265,.85),(1.3,.53,.7)))
    assert not drop_implausible([candidate])
    assert drop_implausible([candidate],upper_margin=.1)
    refined,_ = agree_with_camera([candidate],{1:panel},np.vstack([panel,background]))
    accepted = drop_implausible(refined)
    assert accepted and accepted[0].box.size[0]>1.2 and accepted[0].box.size[1]<.3
    huge = SceneObject(2,'television',box((1.,1.,1.),(4.,2.,2.)))
    assert not drop_implausible([huge],upper_margin=.1)
