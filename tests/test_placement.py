"""Tests for placement checking and search on the synthetic office."""

import numpy as np
import pytest

from conftest import box
from pivot.geometry.collision import polygon_distance
from pivot.geometry.floor import FloorMap
from pivot.planning.placement import ConflictKind, check_placement, propose_placements
from pivot.scene.graph import Scene, SceneObject


def test_current_pose_is_feasible(office: Scene) -> None:
    report = check_placement(office, 3, office[3].box)
    assert report.feasible, report.conflicts
    assert report.travel == 0.0


def test_chair_into_desk_reports_collision(office: Scene) -> None:
    chair = office[3].box
    report = check_placement(office, 3, chair.moved(np.array([1.5, 2.3])))
    kinds = {(c.kind, c.other_id) for c in report.conflicts}
    assert (ConflictKind.COLLISION, 1) in kinds
    conflict = report.to_dict()["conflicts"][0]
    assert conflict["other_label"] == "desk"
    assert conflict["penetration_m"] == pytest.approx(0.45, abs=0.01)


def test_chair_outside_room_reports_off_floor(office: Scene) -> None:
    report = check_placement(office, 3, office[3].box.moved(np.array([-1.0, 1.0])))
    assert [c.kind for c in report.conflicts] == [ConflictKind.OFF_FLOOR]


def test_chair_fits_under_window(office: Scene) -> None:
    # The window starts 1 m above the floor and the chair is 1 m tall.
    report = check_placement(office, 3, office[3].box.moved(np.array([2.8, 3.55])))
    assert report.feasible, report.conflicts


def test_clearance_is_enforced(office: Scene) -> None:
    chair = office[3].box.moved(np.array([1.5, 1.75]))  # 10 cm from the desk
    assert check_placement(office, 3, chair).feasible
    report = check_placement(office, 3, chair, clearance=0.2)
    assert [c.kind for c in report.conflicts] == [ConflictKind.CLEARANCE]


def test_fixed_objects_cannot_move(office: Scene) -> None:
    report = check_placement(office, 5, office[5].box)
    assert [c.kind for c in report.conflicts] == [ConflictKind.NOT_MOVABLE]


def test_desk_move_ignores_its_own_lamp(office: Scene) -> None:
    desk = office[1].box
    report = check_placement(office, 1, desk.moved(desk.center[:2] + [0.5, 0.0]))
    assert report.feasible, report.conflicts


def test_propose_chair_next_to_window(office: Scene) -> None:
    proposals = propose_placements(office, 3, 5, max_results=3)
    assert proposals
    for report in proposals:
        assert check_placement(office, 3, report.box).feasible
        assert polygon_distance(report.box.footprint(), office[5].box.footprint()) <= 0.4
    # Ranked by gap first: the best proposal hugs the window.
    assert proposals[0].gap_to_anchor is not None and proposals[0].gap_to_anchor < 0.1
    centers = [r.box.center[:2] for r in proposals]
    assert all(np.linalg.norm(a - b) >= 0.6 - 1e-9
               for i, a in enumerate(centers) for b in centers[i + 1:])


def test_propose_returns_empty_when_room_is_full() -> None:
    # A 2 x 1 m alcove almost filled by a bed leaves no room for a chair.
    xs, ys = np.meshgrid(np.arange(0.0, 2.0, 0.02), np.arange(0.0, 1.0, 0.02))
    floor = FloorMap.from_points(np.column_stack([xs.ravel(), ys.ravel()]), close_radius=0)
    alcove = Scene.from_objects([
        SceneObject(1, "bed", box((1.0, 0.5, 0.25), (1.8, 0.9, 0.5))),
        SceneObject(2, "chair", box((5.0, 5.0, 0.5), (0.5, 0.5, 1.0))),
    ], floor)
    assert not propose_placements(alcove, 2, 1)


def test_wall_clearance_is_enforced(office: Scene) -> None:
    # The chair 10 cm from the left wall (x = 0).
    chair = office[3].box.moved(np.array([0.4, 1.0]))
    assert check_placement(office, 3, chair).feasible
    report = check_placement(office, 3, chair, clearance=0.2)
    assert [c.kind for c in report.conflicts] == [ConflictKind.WALL_CLEARANCE]
    assert report.conflicts[0].amount == pytest.approx(0.1, abs=0.03)


def test_windows_are_not_obstacles_but_raise_a_warning(office: Scene) -> None:
    # The window (#5) box reaches 5 cm into the room; a chair can stand right in front.
    in_front = office[3].box.moved(np.array([2.8, 3.6]))
    report = check_placement(office, 3, in_front)
    assert report.feasible
    assert report.warnings == ["stands in front of the window (#5)"]
    assert not check_placement(office, 3, office[3].box).warnings


def test_standing_beside_an_opening_is_not_in_front_of_it(office: Scene) -> None:
    # The window (#5) spans x in [2.2, 3.4] on the wall y = 4; the chair is
    # against the same wall, just to its left.
    beside = office[3].box.moved(np.array([1.85, 3.65]))
    report = check_placement(office, 3, beside)
    assert report.feasible
    assert not report.warnings
