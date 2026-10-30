"""How an object can be moved: by how many people, and whether it can be carried.

Masses are unknown, so the estimate comes from what the object is and how big
it is: a sofa or a wardrobe takes two people, a chair one; an unknown object
large enough to be hard to hold alone takes two as well. An object one person
moves can be carried over furniture if it is not too tall (a bin, a chair, a
monitor); a floor lamp is light but too tall, so it slides.
"""

from __future__ import annotations

from pivot.scene.graph import SceneObject

# Heavy or bulky by nature, whatever their measured size.
TWO_PERSON_LABELS = frozenset({
    "sofa", "couch", "wardrobe", "closet", "bookshelf", "bookcase", "bed", "dresser",
    "piano", "refrigerator", "fridge", "washing machine", "armchair", "cabinet", "desk",
})
# Light by nature: one person, even when tall or wide (a floor lamp, a coat rack).
ONE_PERSON_LABELS = frozenset({
    "chair", "office chair", "stool", "lamp", "floor lamp", "desk lamp", "monitor",
    "trash can", "bin", "plant", "box", "carton", "bag", "pillow", "fan", "coat rack",
})
# Beyond these, an object is hard to hold alone.
MAX_ONE_PERSON_SIDE = 1.4
MAX_ONE_PERSON_FOOTPRINT = 0.8
# Taller objects cannot be carried over furniture (a floor lamp, a coat rack).
MAX_CARRIED_HEIGHT = 1.2


def people_needed(obj: SceneObject) -> int:
    """Estimates how many people are needed to move an object safely.

    Args:
        obj: The object.

    Returns:
        1 or 2.
    """
    label = obj.label.lower()
    if label in TWO_PERSON_LABELS:
        return 2
    if label in ONE_PERSON_LABELS:
        return 1
    width, depth = sorted(float(v) for v in obj.box.size[:2])
    large = depth > MAX_ONE_PERSON_SIDE or width * depth > MAX_ONE_PERSON_FOOTPRINT
    return 2 if large else 1


def can_be_carried(obj: SceneObject) -> bool:
    """Tells whether one person can lift an object and carry it over furniture.

    Args:
        obj: The object.

    Returns:
        True for objects one person moves and no taller than
        :data:`MAX_CARRIED_HEIGHT`.
    """
    return people_needed(obj) == 1 and float(obj.box.size[2]) <= MAX_CARRIED_HEIGHT
