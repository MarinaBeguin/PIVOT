"""Rooms: the floor of a home split at its doors, and named after what is in them.

A capture may cover several rooms. On the floor map (floor hidden under
furniture included), the cores of rooms are the cells far enough from any
wall; doorways and narrow passages are not. Each core grows over the floor
until it meets another (a watershed on the distance to the walls), so rooms
meet in their doorways. A room is then named after its objects: a bed makes a
bedroom, an oven a kitchen, a sofa a living room.
"""

from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from pivot.geometry.floor import FloorMap
from pivot.geometry.types import BoolArray, FloatArray, IntArray
from pivot.scene.graph import Scene, SceneObject, is_opening

# Two areas are separate rooms when they meet through a passage (a doorway)
# whose half-width is under this share of the most open point of either side,
# and which is narrower than MAX_DOOR_WIDTH.
NECK_RATIO = 0.6
MAX_DOOR_WIDTH = 1.5
# Open areas less clear of the walls than this do not start a room on their own.
MIN_PEAK_CLEARANCE = 0.3
# Smaller areas are not rooms: they join the neighbor they share most border with.
MIN_ROOM_AREA = 2.0
# What a room contains says what it is, in this order of precedence.
ROOM_KINDS = (
    ("kitchen", ("oven", "stove", "cooker", "refrigerator", "fridge", "dishwasher",
                 "microwave", "kitchen")),
    ("bathroom", ("toilet", "bathtub", "bath tub", "shower")),
    ("bedroom", ("bed",)),
    ("living room", ("sofa", "couch", "tv", "television")),
    ("classroom", ("whiteboard", "blackboard", "projector")),
    ("office", ("desk", "monitor", "computer")),
)
# Below this width (twice the largest distance to the walls), a room is a hallway.
HALLWAY_WIDTH = 1.6
# A room with that many chairs around tables is a dining room.
DINING_CHAIRS = 4


@dataclass(frozen=True)
class Room:
    """One room of the floor.

    Attributes:
        room_id: Label of its cells in :attr:`RoomMap.labels`, from 1.
        name: What it is, e.g. ``"bedroom"``; numbered when several share it.
        area: Floor area, in square meters.
        center: (2,) a point well inside the room (its most open cell).
    """

    room_id: int
    name: str
    area: float
    center: FloatArray


@dataclass(frozen=True)
class Passage:
    """Where two rooms meet: a doorway, or an open passage.

    Attributes:
        rooms: The ids of the two rooms, smaller first.
        center: (2,) its widest point, in meters.
        width: Its width there, in meters.
        door_id: The door object at this passage, if one was segmented.
    """

    rooms: tuple[int, int]
    center: FloatArray
    width: float
    door_id: int | None = None


@dataclass(frozen=True)
class RoomMap:
    """The rooms of a floor map.

    Attributes:
        floor: The floor map the rooms split.
        labels: (rows, cols) room id of each cell, 0 for no room.
        rooms: The rooms, by id order.
        passages: Where rooms meet.
        object_rooms: The room id of each object, 0 if outside every room.
    """

    floor: FloorMap
    labels: IntArray
    rooms: list[Room]
    passages: list[Passage] = field(default_factory=list)
    object_rooms: dict[int, int] = field(default_factory=dict)

    def room_at(self, xy: FloatArray) -> int:
        """Returns the room of a point, 0 if none.

        Args:
            xy: (2,) point, in meters.

        Returns:
            The room id.
        """
        cell = np.floor((np.asarray(xy) - self.floor.origin) / self.floor.resolution)
        col, row = int(cell[0]), int(cell[1])
        if 0 <= row < self.labels.shape[0] and 0 <= col < self.labels.shape[1]:
            return int(self.labels[row, col])
        return 0

    def by_id(self, room_id: int) -> Room | None:
        """Finds a room by id.

        Args:
            room_id: The id.

        Returns:
            The room, or None for 0 or an unknown id.
        """
        return next((r for r in self.rooms if r.room_id == room_id), None)

    def find(self, name: str) -> Room | None:
        """Finds a room by name, case-insensitively; a unique partial name works too.

        Args:
            name: E.g. ``"bedroom"`` or ``"Bedroom 2"``.

        Returns:
            The room, or None if no room or several rooms match.
        """
        query = name.strip().lower()
        exact = [r for r in self.rooms if r.name == query]
        if exact:
            return exact[0]
        partial = [r for r in self.rooms if query in r.name]
        return partial[0] if len(partial) == 1 else None

    def name_of(self, room_id: int) -> str | None:
        """Returns the name of a room, None for 0.

        Args:
            room_id: The id.

        Returns:
            The name.
        """
        room = self.by_id(room_id)
        return room.name if room else None


def _label_rooms(floor: FloorMap) -> tuple[IntArray, FloatArray]:
    """Splits the floor into rooms: a watershed from every open area, then merges.

    Every local maximum of the distance to the walls starts a region; regions
    grow until they meet. Two regions stay apart only where they meet through
    a passage clearly narrower than both (a doorway): its half-width is under
    :data:`NECK_RATIO` times the most open point of either side, and the
    passage is under :data:`MAX_DOOR_WIDTH` wide. All others merge.

    Args:
        floor: The floor map.

    Returns:
        The room label of each cell (0 off the floor), and the distance of
        each cell to the nearest wall, in meters.
    """
    clearance = ndimage.distance_transform_edt(np.pad(floor.floor, 1))[1:-1, 1:-1]
    clearance = clearance * floor.resolution
    peaks = (clearance == ndimage.maximum_filter(clearance, size=5)) & floor.floor
    peaks &= clearance >= MIN_PEAK_CLEARANCE
    markers, count = ndimage.label(peaks)
    if count == 0:
        return floor.floor.astype(np.int64), clearance
    labels = _flood(markers.astype(np.int64), clearance, floor.floor)
    return _merge_regions(labels, clearance), clearance


def _flood(markers: IntArray, clearance: FloatArray, mask: BoolArray) -> IntArray:
    """Grows labeled seeds over a mask, the most open cells first (a watershed).

    Each cell, taken by decreasing distance to the walls, joins the region of
    its first labeled neighbor, so regions meet where the floor is narrowest.

    Args:
        markers: Seed labels, 0 elsewhere.
        clearance: Distance of each cell to the nearest wall, in meters.
        mask: Cells to label.

    Returns:
        The labels; 0 for cells off the mask or not connected to any seed.
    """
    labels = np.where(mask, markers, 0)
    rows, cols = labels.shape
    queue = [(-float(clearance[r, c]), int(r), int(c)) for r, c in zip(*np.nonzero(labels))]
    heapq.heapify(queue)
    while queue:
        _, r, c = heapq.heappop(queue)
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and labels[nr, nc] == 0:
                labels[nr, nc] = labels[r, c]
                heapq.heappush(queue, (-float(clearance[nr, nc]), nr, nc))
    return labels


def _saddles(labels: IntArray, clearance: FloatArray) -> dict[tuple[int, int], float]:
    """Finds, for each pair of touching regions, the widest point of their border.

    Args:
        labels: Region labels.
        clearance: Distance of each cell to the nearest wall, in meters.

    Returns:
        The half-width of the passage between each pair, in meters.
    """
    saddles: dict[tuple[int, int], float] = {}
    for dr, dc in ((0, 1), (1, 0)):
        a = labels[: labels.shape[0] - dr, : labels.shape[1] - dc]
        b = labels[dr:, dc:]
        ca = clearance[: labels.shape[0] - dr, : labels.shape[1] - dc]
        cb = clearance[dr:, dc:]
        touching = (a > 0) & (b > 0) & (a != b)
        for x, y, value in zip(a[touching].tolist(), b[touching].tolist(),
                               np.minimum(ca, cb)[touching].tolist()):
            pair = (min(x, y), max(x, y))
            saddles[pair] = max(saddles.get(pair, 0.0), value)
    return saddles


def _merge_regions(labels: IntArray, clearance: FloatArray) -> IntArray:
    """Merges the regions that do not meet through a doorway (see :func:`_label_rooms`).

    Args:
        labels: Region labels from the watershed.
        clearance: Distance of each cell to the nearest wall, in meters.

    Returns:
        The merged labels.
    """
    labels = labels.copy()
    peak = {int(i): float(v) for i, v in zip(
        np.unique(labels[labels > 0]),
        ndimage.maximum(clearance, labels, index=np.unique(labels[labels > 0])))}
    saddles = _saddles(labels, clearance)
    while saddles:
        (a, b), value = max(saddles.items(),
                            key=lambda item: item[1] / min(peak[item[0][0]], peak[item[0][1]]))
        if value < NECK_RATIO * min(peak[a], peak[b]) and 2.0 * value < MAX_DOOR_WIDTH:
            break
        labels[labels == b] = a
        peak[a] = max(peak[a], peak.pop(b))
        merged: dict[tuple[int, int], float] = {}
        for (x, y), v in saddles.items():
            x, y = (a if x == b else x), (a if y == b else y)
            if x != y:
                key = (min(x, y), max(x, y))
                merged[key] = max(merged.get(key, 0.0), v)
        saddles = merged
    return labels


def _merge_small(labels: IntArray, floor: FloorMap) -> IntArray:
    """Merges rooms smaller than :data:`MIN_ROOM_AREA` into their main neighbor.

    Args:
        labels: Room labels.
        floor: The floor map.

    Returns:
        Labels renumbered from 1, without small rooms.
    """
    labels = labels.copy()
    cell_area = floor.resolution ** 2
    while True:
        ids, counts = np.unique(labels[labels > 0], return_counts=True)
        small = [i for i, c in zip(ids, counts) if c * cell_area < MIN_ROOM_AREA]
        if not small or len(ids) <= 1:
            break
        room = small[0]
        ring = ndimage.binary_dilation(labels == room) & (labels != room) & (labels > 0)
        neighbors = Counter(labels[ring].tolist())
        labels[labels == room] = neighbors.most_common(1)[0][0] if neighbors else 0
    renumbered = np.zeros_like(labels)
    for new, old in enumerate(np.unique(labels[labels > 0]), start=1):
        renumbered[labels == old] = new
    return renumbered


def _passages(labels: IntArray, clearance: FloatArray, floor: FloorMap,
              doors: list[SceneObject]) -> list[Passage]:
    """Finds where rooms meet, and the door there if there is one.

    Args:
        labels: Room labels.
        clearance: Distance of each cell to the nearest wall, in meters.
        floor: The floor map.
        doors: Door objects.

    Returns:
        One passage per pair of adjacent rooms.
    """
    passages = []
    for pair, (half_width, cell) in sorted(_widest_border_cells(labels, clearance).items()):
        center = _cell_center(floor, cell)
        near = [(float(np.linalg.norm(d.box.center[:2] - center)), d.object_id) for d in doors]
        distance, door_id = min(near, default=(np.inf, None))
        passages.append(Passage(pair, center, 2.0 * half_width,
                                door_id if distance <= 1.0 else None))
    return passages


def _widest_border_cells(labels: IntArray, clearance: FloatArray
                         ) -> dict[tuple[int, int], tuple[float, tuple[int, int]]]:
    """Finds, for each pair of touching rooms, the most open cell of their border.

    Args:
        labels: Room labels.
        clearance: Distance of each cell to the nearest wall, in meters.

    Returns:
        For each pair (smaller id first), the cell's clearance and ``(row, col)``.
    """
    best: dict[tuple[int, int], tuple[float, tuple[int, int]]] = {}
    for dr, dc in ((0, 1), (1, 0)):
        a = labels[: labels.shape[0] - dr, : labels.shape[1] - dc]
        b = labels[dr:, dc:]
        rows, cols = np.nonzero((a > 0) & (b > 0) & (a != b))
        for r, c in zip(rows.tolist(), cols.tolist()):
            pair = (min(int(a[r, c]), int(b[r, c])), max(int(a[r, c]), int(b[r, c])))
            if pair not in best or clearance[r, c] > best[pair][0]:
                best[pair] = (float(clearance[r, c]), (r, c))
    return best


def _cell_center(floor: FloorMap, cell: tuple[int, int]) -> FloatArray:
    """Returns the world position of a floor map cell's center.

    Args:
        floor: The floor map.
        cell: ``(row, col)``.

    Returns:
        (2,) point, in meters.
    """
    center: FloatArray = floor.origin + (np.array([cell[1], cell[0]]) + 0.5) * floor.resolution
    return center


def _object_rooms(scene: Scene, labels: IntArray) -> dict[int, int]:
    """Puts each object in the room under its center, or the nearest one.

    Args:
        scene: The scene.
        labels: Room labels of its floor map.

    Returns:
        The room id of each object, 0 if no room is within half a meter.
    """
    floor = scene.floor
    assert floor is not None
    _, (near_rows, near_cols) = ndimage.distance_transform_edt(labels == 0, return_indices=True)
    reach = 0.5 / floor.resolution
    result = {}
    for obj in scene.objects.values():
        col, row = np.floor((obj.box.center[:2] - floor.origin) / floor.resolution).astype(int)
        row = min(max(row, 0), labels.shape[0] - 1)
        col = min(max(col, 0), labels.shape[1] - 1)
        nr, nc = near_rows[row, col], near_cols[row, col]
        close = np.hypot(nr - row, nc - col) <= reach
        result[obj.object_id] = int(labels[nr, nc]) if close else 0
    return result


def _kind(objects: list[SceneObject], widest: float) -> str:
    """Says what kind of room holds these objects.

    Args:
        objects: The objects of the room.
        widest: Twice the largest distance to the walls in the room, in meters.

    Returns:
        A room kind, e.g. ``"kitchen"``.
    """
    labels = [o.label.lower() for o in objects if not is_opening(o)]
    for kind, words in ROOM_KINDS:
        if any(any(w in label.split() or label == w for w in words) for label in labels):
            return kind
    chairs = sum("chair" in label for label in labels)
    if chairs >= DINING_CHAIRS and any("table" in label for label in labels):
        return "dining room"
    return "hallway" if widest < HALLWAY_WIDTH else "room"


def find_rooms(scene: Scene) -> RoomMap | None:
    """Splits the floor of a scene into named rooms.

    Args:
        scene: The scene; needs a floor map.

    Returns:
        The rooms, or None if the scene has no floor map.
    """
    floor = scene.floor
    if floor is None:
        return None
    labels, clearance = _label_rooms(floor)
    labels = _merge_small(labels, floor)
    object_rooms = _object_rooms(scene, labels)
    kinds = {}
    for room_id in range(1, int(labels.max()) + 1):
        inside = [scene[i] for i, r in object_rooms.items() if r == room_id]
        kinds[room_id] = _kind(inside, 2.0 * float(clearance[labels == room_id].max()))
    doors = [o for o in scene.objects.values() if o.label.lower() == "door"]
    return RoomMap(floor, labels, _named_rooms(kinds, labels, clearance, floor),
                   _passages(labels, clearance, floor, doors), object_rooms)


def _named_rooms(kinds: dict[int, str], labels: IntArray, clearance: FloatArray,
                 floor: FloorMap) -> list[Room]:
    """Names the rooms after their kinds, numbering the kinds found several times.

    Args:
        kinds: The kind of each room, by id.
        labels: Room labels.
        clearance: Distance of each cell to the nearest wall, in meters.
        floor: The floor map.

    Returns:
        The rooms, by id order.
    """
    counts = Counter(kinds.values())
    seen: Counter[str] = Counter()
    rooms = []
    for room_id, kind in kinds.items():
        seen[kind] += 1
        cells = labels == room_id
        index = int(np.argmax(np.where(cells, clearance, -1.0)))
        center = _cell_center(floor, (index // cells.shape[1], index % cells.shape[1]))
        rooms.append(Room(room_id, f"{kind} {seen[kind]}" if counts[kind] > 1 else kind,
                          float(cells.sum()) * floor.resolution ** 2, center))
    return rooms
