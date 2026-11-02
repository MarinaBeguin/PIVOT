"""Dropping objects bigger or smaller than any real object of their label.

A mask that leaks onto a wall or the floor, or that follows a reflection,
lifts to a box meters wide: a dresser of 3 x 3 x 2.5 m, a rug half a meter
thick. No object of the label is that big, so the object is a false
detection and is dropped rather than left to block the room. A mask on a
small part of a big piece of furniture, or on something else, lifts to a
box too small for its label: a "bed" 27 cm wide along the real bed, a
"desk" 50 cm high that is a bin. It is dropped too. The limits are
generous: real furniture stays well inside them.
"""

from __future__ import annotations

from pivot.scene.graph import SceneObject

# Largest size of one object of a label: long side, short side and height, in meters.
# Tables joined in a row (a classroom, a meeting room) are seen as one long table.
MAX_SIZE = {
    "chair": (1.5, 1.2, 1.5), "office chair": (1.0, 1.0, 1.5), "armchair": (1.4, 1.2, 1.3),
    "stool": (0.7, 0.7, 1.1), "sofa": (3.6, 2.2, 1.3), "bed": (2.6, 2.4, 1.8),
    "table": (6.0, 1.8, 1.2), "coffee table": (1.8, 1.2, 0.7), "desk": (2.6, 1.4, 1.3),
    "dresser": (2.2, 0.9, 1.8), "nightstand": (0.9, 0.8, 1.0), "tv stand": (2.6, 0.9, 1.1),
    "cabinet": (2.6, 1.0, 2.5), "wardrobe": (2.8, 1.0, 2.7), "shelf": (2.8, 0.9, 2.7),
    "bookshelf": (2.8, 0.9, 2.7), "television": (2.2, 0.5, 1.5), "monitor": (1.2, 0.5, 0.9),
    "trash can": (0.7, 0.7, 1.1), "rug": (4.5, 3.5, 0.15), "plant": (1.6, 1.6, 2.6),
}

# Smallest size of one object of a label, for big furniture: long side, short side
# and height, in meters.
MIN_SIZE = {
    "bed": (1.2, 0.7, 0.2), "sofa": (1.0, 0.6, 0.35), "desk": (0.6, 0.4, 0.6),
    "table": (0.4, 0.3, 0.3), "wardrobe": (0.6, 0.35, 1.2), "dresser": (0.5, 0.3, 0.5),
}


def plausible(obj: SceneObject) -> bool:
    """Tests whether an object is neither bigger nor smaller than real objects of its label.

    Args:
        obj: The object.

    Returns:
        True if its box is within the limits of its label (those it has).
    """
    long_side, short_side = sorted(obj.box.size[:2], reverse=True)
    size = (long_side, short_side, float(obj.box.size[2]))
    largest = MAX_SIZE.get(obj.label.lower())
    smallest = MIN_SIZE.get(obj.label.lower())
    return bool((largest is None or all(s <= m for s, m in zip(size, largest)))
                and (smallest is None or all(s >= m for s, m in zip(size, smallest))))


def drop_implausible(objects: list[SceneObject]) -> list[SceneObject]:
    """Drops objects bigger or smaller than any real object of their label.

    Args:
        objects: Objects with settled boxes.

    Returns:
        The plausible objects.
    """
    return [o for o in objects if plausible(o)]
