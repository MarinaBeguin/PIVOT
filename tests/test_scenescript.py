"""Tests of the bridge to SceneScript (without the model)."""

import gzip
from pathlib import Path

import numpy as np
import pytest

from pivot.lifting.scenescript import (
    openings_from_program, parse_program, with_scenescript_openings, write_points,
)
from pivot.scene.examples import office_scene

PROGRAM = "\n".join([
    "make_wall, id=0, a_x=0.0, a_y=4.0, a_z=0.0, b_x=5.0, b_y=4.0, b_z=0.0, height=2.6, "
    "thickness=0.0",
    "make_wall, id=1, a_x=0.0, a_y=0.0, a_z=0.0, b_x=0.0, b_y=4.0, b_z=0.0, height=2.6, "
    "thickness=0.12",
    "make_door, id=1000, wall0_id=1, wall1_id=1, position_x=0.0, position_y=1.0, "
    "position_z=1.0, width=0.9, height=2.0",
    "make_window, id=2000, wall0_id=0, wall1_id=0, position_x=2.5, position_y=4.0, "
    "position_z=1.5, width=1.2, height=1.1",
    "make_bbox, id=3000, class=chair, position_x=1.0, position_y=1.0, position_z=0.4"])


def test_the_program_becomes_doors_and_windows_set_in_their_walls() -> None:
    assert [e["command"] for e in parse_program(PROGRAM)] == [
        "make_wall", "make_wall", "make_door", "make_window", "make_bbox"]
    openings = openings_from_program(PROGRAM, first_id=50)
    assert len(openings) == 2
    door, window = openings[0], openings[1]
    assert (door.object_id, door.label, door.movable) == (50, "door", False)
    assert np.allclose(door.box.size, [0.9, 0.12, 2.0])
    assert door.box.yaw == pytest.approx(np.pi / 2)  # Along its wall, which runs along y.
    assert (window.label, window.box.z_min) == ("window", pytest.approx(0.95))
    assert window.box.size[1] == 0.1  # A wall without thickness.


def test_scenescript_openings_replace_the_scene_s_own() -> None:
    scene = with_scenescript_openings(office_scene(), PROGRAM)
    labels = sorted(o.label for o in scene.objects.values())
    assert labels.count("window") == 1 and labels.count("door") == 1
    assert scene.metadata["openings"] == "SceneScript"
    unchanged = office_scene()
    assert with_scenescript_openings(unchanged, "") is unchanged


def test_points_are_written_as_aria_semi_dense_points(tmp_path: Path) -> None:
    write_points(np.array([[1.0, 2.0, 3.0]]), tmp_path / "points.csv.gz")
    with gzip.open(tmp_path / "points.csv.gz", "rt") as lines:
        assert lines.read().splitlines() == [
            "uid,graph_uid,px_world,py_world,pz_world,inv_dist_std,dist_std",
            "0,0,1.0000,2.0000,3.0000,0,0"]
