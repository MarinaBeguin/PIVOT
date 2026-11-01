"""Dropping objects bigger than any real object of their label.

A mask that leaks onto a wall or the floor, or that follows a reflection,
lifts to a box meters wide: a dresser of 3 x 3 x 2.5 m, a rug half a meter
thick. No object of the label is that big, so the object is a false
detection and is dropped rather than left to block the room. The limits are
generous: real furniture stays well inside them.
"""

from __future__ import annotations

from pivot.scene.graph import SceneObject

# Largest size of one object of a label: long side, short side and height, in meters.
MAX_SIZE = {
    "chair": (1.5, 1.2, 1.5), "office chair": (1.0, 1.0, 1.5), "armchair": (1.4, 1.2, 1.3),
    "stool": (0.7, 0.7, 1.1), "sofa": (3.6, 2.2, 1.3), "bed": (2.6, 2.4, 1.8),
    "table": (3.2, 1.8, 1.2), "coffee table": (1.8, 1.2, 0.7), "desk": (2.6, 1.4, 1.3),
    "dresser": (2.2, 0.9, 1.8), "nightstand": (0.9, 0.8, 1.0), "tv stand": (2.6, 0.9, 1.1),
    "cabinet": (2.6, 1.0, 2.5), "wardrobe": (2.8, 1.0, 2.7), "shelf": (2.8, 0.9, 2.7),
    "bookshelf": (2.8, 0.9, 2.7), "television": (2.2, 0.5, 1.5), "monitor": (1.2, 0.5, 1.4),
    "trash can": (0.7, 0.7, 1.1), "rug": (4.5, 3.5, 0.15), "plant": (1.6, 1.6, 2.6),
}


def plausible(obj: SceneObject) -> bool:
    """Tests whether an object is no bigger than real objects of its label.

    Args:
        obj: The object.

    Returns:
        True if its label has no known limit or its box is within it.
    """
    limit = MAX_SIZE.get(obj.label.lower())
    if limit is None:
        return True
    long_side, short_side = sorted(obj.box.size[:2], reverse=True)
    return bool(long_side <= limit[0] and short_side <= limit[1] and obj.box.size[2] <= limit[2])


def drop_implausible(objects: list[SceneObject]) -> list[SceneObject]:
    """Drops objects bigger than any real object of their label.

    Args:
        objects: Objects with settled boxes.

    Returns:
        The plausible objects.
    """
    return [o for o in objects if plausible(o)]
