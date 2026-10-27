"""Tests for the scene graph."""

from pathlib import Path

import numpy as np
import pytest

from conftest import box
from pivot.scene.graph import Scene, SceneObject, carried_objects, support_parents


def test_find_is_case_insensitive(office: Scene) -> None:
    assert [o.object_id for o in office.find("CHAIR")] == [3]
    assert office.find("sofa") == []


def test_missing_object_raises_key_error(office: Scene) -> None:
    with pytest.raises(KeyError, match="42"):
        _ = office[42]


def test_duplicate_ids_are_rejected(office: Scene) -> None:
    desk = office[1]
    with pytest.raises(ValueError):
        Scene.from_objects([desk, desk])


def test_lamp_rests_on_desk(office: Scene) -> None:
    assert support_parents(office) == {2: 1}
    assert carried_objects(office, 1) == [2]
    assert not carried_objects(office, 3)


def test_moving_desk_carries_lamp(office: Scene) -> None:
    desk = office[1].box
    moved = office.moved(1, desk.moved(desk.center[:2] + [1.0, 0.0], yaw=desk.yaw + np.pi / 2))
    # The lamp keeps its position relative to the desk: 30 cm left of center,
    # which becomes 30 cm "down" after a quarter turn counter-clockwise.
    np.testing.assert_allclose(moved[2].box.center[:2], [2.4, 2.2], atol=1e-9)
    np.testing.assert_allclose(moved[2].box.z_min, office[2].box.z_min)
    # The original scene is untouched.
    np.testing.assert_allclose(office[2].box.center, [1.2, 2.6, 0.95])


def test_round_trip_through_json(office: Scene, tmp_path: Path) -> None:
    path = tmp_path / "scene.json"
    office.save(path)
    restored = Scene.load(path)
    assert set(restored.objects) == set(office.objects)
    assert restored[5].movable is False
    assert restored.floor is not None and office.floor is not None
    np.testing.assert_array_equal(restored.floor.floor, office.floor.floor)


def test_object_serialization_keeps_label() -> None:
    obj = SceneObject(7, "plant", box((0.0, 0.0, 0.3), (0.3, 0.3, 0.6)), confidence=0.8)
    restored = SceneObject.from_dict(obj.to_dict())
    assert restored.label == "plant"
    assert restored.confidence == pytest.approx(0.8)
