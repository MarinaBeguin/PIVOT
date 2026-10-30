"""Scene graph: the objects of a room and where they are.

Scenes are immutable. Moving an object returns a new scene, which gives the
agent cheap what-if reasoning and the viewer a free undo history.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from pivot.geometry.box import OrientedBox
from pivot.geometry.collision import polygon_penetration
from pivot.geometry.floor import FloorMap
from pivot.geometry.shape import VoxelShape

# Labels of objects that are part of the building and can never be moved.
STRUCTURAL_LABELS = frozenset({"wall", "floor", "ceiling", "door", "window", "radiator", "column"})

# Openings in walls: not obstacles inside the room (their volume is mostly what
# the camera saw through them), but places that should be kept clear.
OPENING_LABELS = frozenset({"door", "window"})

# Furniture whose top can receive an object put down on it (a trash can or a
# chair is not one, even if its top is flat enough).
SURFACE_LABELS = frozenset({"table", "desk", "shelf", "cabinet", "counter", "nightstand",
                            "dresser", "sideboard", "bookshelf", "bench", "tv stand",
                            "coffee table"})


def is_opening(obj: SceneObject) -> bool:
    """Tests whether an object is a door or a window.

    Args:
        obj: The object.

    Returns:
        True for openings in walls.
    """
    return obj.label.lower() in OPENING_LABELS


@dataclass(frozen=True)
class SceneObject:
    """One object instance in the scene.

    Attributes:
        object_id: Unique identifier, stable across scene edits.
        label: Short noun phrase, e.g. ``"office chair"``.
        box: Gravity-aligned bounding box in world coordinates.
        movable: Whether the agent may move the object.
        confidence: Detection confidence in [0, 1].
        shape: Volume actually occupied inside the box, if known; the whole
            box otherwise.
    """

    object_id: int
    label: str
    box: OrientedBox
    movable: bool = True
    confidence: float = 1.0
    shape: VoxelShape | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serializes the object to JSON-compatible types.

        Returns:
            A dictionary representation.
        """
        data: dict[str, Any] = {
            "id": self.object_id,
            "label": self.label,
            "box": self.box.to_dict(),
            "movable": self.movable,
            "confidence": self.confidence,
        }
        if self.shape is not None:
            data["shape"] = self.shape.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneObject:
        """Builds an object from the output of :meth:`to_dict`.

        Args:
            data: Serialized object.

        Returns:
            The deserialized object.
        """
        shape = VoxelShape.from_dict(data["shape"]) if data.get("shape") else None
        return cls(int(data["id"]), str(data["label"]), OrientedBox.from_dict(data["box"]),
                   bool(data["movable"]), float(data["confidence"]), shape)


@dataclass(frozen=True)
class Scene:
    """A room: its objects, its walkable floor and its ceiling.

    Attributes:
        objects: Objects keyed by id.
        floor: Walkable floor map, or None if unknown.
        metadata: Free-form provenance information (source video, models used).
        ceiling: Ceiling height above the floor, in meters, or None if unknown.
    """

    objects: dict[int, SceneObject]
    floor: FloorMap | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    ceiling: float | None = None

    @classmethod
    def from_objects(cls, objects: list[SceneObject], floor: FloorMap | None = None,
                     metadata: dict[str, Any] | None = None,
                     ceiling: float | None = None) -> Scene:
        """Builds a scene from a list of objects.

        Args:
            objects: Objects with unique ids.
            floor: Walkable floor map.
            metadata: Provenance information.
            ceiling: Ceiling height, in meters.

        Returns:
            The scene.

        Raises:
            ValueError: If two objects share an id.
        """
        by_id = {obj.object_id: obj for obj in objects}
        if len(by_id) != len(objects):
            raise ValueError("Object ids must be unique.")
        return cls(by_id, floor, dict(metadata or {}), ceiling)

    def __getitem__(self, object_id: int) -> SceneObject:
        try:
            return self.objects[object_id]
        except KeyError:
            raise KeyError(f"No object with id {object_id} in the scene.") from None

    def __len__(self) -> int:
        return len(self.objects)

    def find(self, query: str) -> list[SceneObject]:
        """Finds objects whose label contains a query, case-insensitively.

        Args:
            query: Substring to search for, e.g. ``"chair"``.

        Returns:
            Matching objects, sorted by id.
        """
        query = query.strip().lower()
        return sorted((o for o in self.objects.values() if query in o.label.lower()),
                      key=lambda o: o.object_id)

    def with_box(self, object_id: int, box: OrientedBox) -> Scene:
        """Returns a copy of the scene with one object's box replaced.

        Args:
            object_id: Object to update.
            box: Its new box.

        Returns:
            The edited scene.
        """
        objects = dict(self.objects)
        objects[object_id] = dataclasses.replace(self[object_id], box=box)
        return dataclasses.replace(self, objects=objects)

    def moved(self, object_id: int, box: OrientedBox) -> Scene:
        """Moves an object, carrying along everything resting on it.

        Supported objects undergo the same rigid motion (rotation about the
        vertical axis through the mover's center, then translation).

        Args:
            object_id: Object to move.
            box: Its new box. Must have the same size as the current box.

        Returns:
            The edited scene.
        """
        scene = self.with_box(object_id, box)
        for child_id, child_box in carried_boxes(self, object_id, box).items():
            scene = scene.with_box(child_id, child_box)
        return scene

    def to_dict(self) -> dict[str, Any]:
        """Serializes the scene to JSON-compatible types.

        Returns:
            A dictionary representation.
        """
        return {
            "objects": [o.to_dict() for o in sorted(self.objects.values(),
                                                     key=lambda o: o.object_id)],
            "floor": self.floor.to_dict() if self.floor is not None else None,
            "ceiling": self.ceiling,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        """Builds a scene from the output of :meth:`to_dict`.

        Args:
            data: Serialized scene.

        Returns:
            The deserialized scene.
        """
        floor = FloorMap.from_dict(data["floor"]) if data.get("floor") else None
        objects = [SceneObject.from_dict(o) for o in data["objects"]]
        return cls.from_objects(objects, floor, data.get("metadata"), data.get("ceiling"))

    def save(self, path: Path) -> None:
        """Writes the scene to a JSON file.

        Args:
            path: Destination file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Scene:
        """Reads a scene written by :meth:`save`.

        Args:
            path: Source file.

        Returns:
            The scene.
        """
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def is_supported_by(top: OrientedBox, bottom: OrientedBox, height_tolerance: float = 0.05,
                    min_overlap: float = 0.5) -> bool:
    """Tests whether one box rests on top of another.

    Args:
        top: Candidate supported box.
        bottom: Candidate supporting box.
        height_tolerance: Allowed vertical gap or interpenetration between the
            bottom of ``top`` and the top of ``bottom``, in meters.
        min_overlap: Minimum fraction of ``top``'s footprint that must lie over
            ``bottom``'s footprint, approximated by penetration depth along
            ``top``'s smaller side.

    Returns:
        True if ``top`` rests on ``bottom``.
    """
    if abs(top.z_min - bottom.z_max) > height_tolerance:
        return False
    depth = polygon_penetration(top.footprint(), bottom.footprint())
    return depth >= min_overlap * float(min(top.size[0], top.size[1]))


def support_parents(scene: Scene) -> dict[int, int]:
    """Finds, for every object resting on another object, its support.

    When several supports qualify, the highest one is chosen.

    Args:
        scene: The scene.

    Returns:
        A mapping from supported object id to supporting object id.
    """
    parents: dict[int, int] = {}
    for top in scene.objects.values():
        supports = [b for b in scene.objects.values()
                    if b.object_id != top.object_id and is_supported_by(top.box, b.box)]
        if supports:
            parents[top.object_id] = max(supports, key=lambda b: b.box.z_max).object_id
    return parents


def carried_objects(scene: Scene, object_id: int) -> list[int]:
    """Lists the objects that move along with an object, transitively.

    Args:
        scene: The scene.
        object_id: The object being moved.

    Returns:
        Ids of objects stacked on it, directly or indirectly, in breadth-first order.
    """
    parents = support_parents(scene)
    carried: list[int] = []
    frontier = [object_id]
    while frontier:
        current = frontier.pop(0)
        children = sorted(c for c, p in parents.items() if p == current and c not in carried)
        carried.extend(children)
        frontier.extend(children)
    return carried


def carried_boxes(scene: Scene, object_id: int, box: OrientedBox) -> dict[int, OrientedBox]:
    """Places the objects carried by an object at a new pose of that object.

    Carried objects undergo the same rigid motion: rotation about the vertical
    axis through the carrier's center, then translation.

    Args:
        scene: The scene.
        object_id: The carrier.
        box: Its new box.

    Returns:
        The new boxes of the carried objects, keyed by id.
    """
    old = scene[object_id].box
    delta_yaw = box.yaw - old.yaw
    rotation = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)],
                         [np.sin(delta_yaw), np.cos(delta_yaw)]])
    delta_z = box.center[2] - old.center[2]
    result = {}
    for child_id in carried_objects(scene, object_id):
        child = scene[child_id].box
        center_xy = rotation @ (child.center[:2] - old.center[:2]) + box.center[:2]
        result[child_id] = child.moved(center_xy, child.yaw + delta_yaw, child.z_min + delta_z)
    return result
