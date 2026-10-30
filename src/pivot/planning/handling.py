"""How many people it takes to move an object.

Masses are unknown, so the estimate comes from what the object is and how big
it is: a sofa or a wardrobe takes two people, a chair one; an unknown object
large enough to be hard to hold alone takes two as well.
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
