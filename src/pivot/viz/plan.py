"""Printable moving instructions: a plan seen from above and the numbered steps.

The plan shows the floor, the objects where they end up, each moved object's
start (dashed) and end (filled) joined by a numbered arrow, and the floor a
wheelchair can still reach from the doors after the moves.
"""

from __future__ import annotations

import html

import numpy as np

from pivot.geometry.types import BoolArray, FloatArray
from pivot.planning.access import WALKING_WIDTH, WHEELCHAIR_WIDTH, AccessReport, check_access
from pivot.planning.instructions import Step
from pivot.scene.graph import Scene, is_opening
from pivot.scene.rooms import find_rooms

# Opens the print dialog once the page is shown; "Save as PDF" is one of its printers.
SAVE_PDF_ON_LOAD = " onload='print()'"
_WIDTH_PX = 640.0
_PAD_PX = 16.0
_STYLE = """
body { font: 15px/1.45 system-ui, sans-serif; color: #1d2421; background: #fff;
       max-width: 720px; margin: 24px auto; padding: 0 16px; }
h1 { font-size: 22px; margin: 0 0 4px; } .muted { color: #5b6661; font-size: 13px; }
ol { padding-left: 22px; } li { margin: 6px 0; }
.access { border-left: 4px solid #2f9e6b; padding: 6px 10px; background: #eef7f2; }
.access.no { border-color: #c2412d; background: #fbeeeb; }
svg { width: 100%; height: auto; border: 1px solid #d5dbd8; border-radius: 6px; }
button { font: inherit; padding: 6px 12px; margin: 10px 0; }
@media print { button { display: none; } }
"""


class _Frame:
    """Maps world xy (meters, y up) to SVG coordinates (pixels, y down).

    Args:
        low: (2,) lowest world xy shown.
        high: (2,) highest world xy shown.
    """

    def __init__(self, low: FloatArray, high: FloatArray):
        self.low, self.high = low, high
        self.scale = (_WIDTH_PX - 2 * _PAD_PX) / max(float(high[0] - low[0]), 1e-6)
        self.height = float(high[1] - low[1]) * self.scale + 2 * _PAD_PX

    def point(self, xy: FloatArray) -> tuple[float, float]:
        """Converts a world point.

        Args:
            xy: (2,) world point.

        Returns:
            The SVG point.
        """
        return (_PAD_PX + (float(xy[0]) - float(self.low[0])) * self.scale,
                _PAD_PX + (float(self.high[1]) - float(xy[1])) * self.scale)

    def polygon(self, corners: FloatArray) -> str:
        """Converts a polygon to an SVG ``points`` attribute.

        Args:
            corners: (N, 2) world polygon.

        Returns:
            The attribute value.
        """
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in map(self.point, corners))


def _cell_runs(scene: Scene, mask: BoolArray, frame: _Frame, fill: str, opacity: float) -> str:
    """Draws the cells of a floor map mask as horizontal runs of rectangles.

    Args:
        scene: The scene, holding the floor map.
        mask: (rows, cols) cells to draw.
        frame: Coordinate mapping.
        fill: Fill color.
        opacity: Fill opacity.

    Returns:
        SVG elements.
    """
    floor = scene.floor
    assert floor is not None
    size = floor.resolution * frame.scale
    shapes = []
    for row in range(mask.shape[0]):
        cols = np.flatnonzero(mask[row])
        if len(cols) == 0:
            continue
        breaks = np.flatnonzero(np.diff(cols) > 1)
        for first, last in zip(np.r_[cols[0], cols[breaks + 1]], np.r_[cols[breaks], cols[-1]]):
            x, y = frame.point(floor.origin + np.array([first, row + 1]) * floor.resolution)
            shapes.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{(last - first + 1) * size:.1f}" '
                          f'height="{size + 0.3:.1f}" fill="{fill}" fill-opacity="{opacity}"/>')
    return "".join(shapes)


def _frame(final: Scene, steps: list[Step]) -> _Frame:
    """Fits the drawing to the floor, the objects and the starts of the steps.

    Args:
        final: The scene after every step.
        steps: The steps.

    Returns:
        The coordinate mapping.
    """
    corners = [o.box.footprint() for o in final.objects.values()]
    corners += [s.start.footprint() for s in steps]
    if final.floor is not None:
        rows, cols = np.nonzero(final.floor.floor)
        cells = np.column_stack([[cols.min(), cols.max() + 1], [rows.min(), rows.max() + 1]])
        corners.append(final.floor.origin + cells * final.floor.resolution)
    points = np.vstack(corners)
    return _Frame(points.min(axis=0) - 0.1, points.max(axis=0) + 0.1)


def _rooms_svg(final: Scene, frame: _Frame) -> list[str]:
    """Writes the name of each room on the plan, when there are several.

    Args:
        final: The scene.
        frame: Coordinate mapping.

    Returns:
        SVG elements.
    """
    rooms = find_rooms(final)
    if rooms is None or len(rooms.rooms) < 2:
        return []
    labels = []
    for room in rooms.rooms:
        x, y = frame.point(room.center)
        labels.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="middle" font-size="14" '
                      f'font-weight="600" fill="#5b6661">{html.escape(room.name)}</text>')
    return labels


def _objects_svg(final: Scene, frame: _Frame, moved: set[int]) -> list[str]:
    """Draws the objects where they end up, moved ones highlighted, numbered.

    Args:
        final: The scene after every step.
        frame: Coordinate mapping.
        moved: Ids of the moved objects.

    Returns:
        SVG elements.
    """
    shapes, numbers = [], []
    for obj in sorted(final.objects.values(), key=lambda o: o.box.z_max):
        corners = frame.polygon(obj.box.footprint())
        if is_opening(obj):
            color = "#8a5a2b" if obj.label.lower() == "door" else "#3d7fc1"
            shapes.append(f'<polygon points="{corners}" fill="{color}"/>')
            continue
        fill = "#f2b84b" if obj.object_id in moved else "#aab4b0"
        shapes.append(f'<polygon points="{corners}" fill="{fill}" fill-opacity="0.85" '
                      'stroke="#4b5652" stroke-width="0.8"/>')
        x, y = frame.point(obj.box.center[:2])
        # Numbers go last, so that no object drawn later hides them.
        numbers.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" fill="#1d2421" '
                       f'stroke="#fff" stroke-width="2.5" paint-order="stroke">'
                       f'{obj.object_id}</text>')
    return shapes + numbers


def _steps_svg(steps: list[Step], frame: _Frame) -> list[str]:
    """Draws each step: dashed start, arrow to the end, step number.

    Args:
        steps: The steps.
        frame: Coordinate mapping.

    Returns:
        SVG elements.
    """
    parts = []
    for step in steps:
        parts.append(f'<polygon points="{frame.polygon(step.start.footprint())}" fill="none" '
                     'stroke="#c2412d" stroke-width="1.2" stroke-dasharray="4 3"/>')
        (x0, y0), (x1, y1) = frame.point(step.start.center[:2]), frame.point(step.end.center[:2])
        parts.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" '
                     'stroke="#c2412d" stroke-width="2" marker-end="url(#arrow)"/>')
        parts.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="9" fill="#c2412d"/>'
                     f'<text x="{x0:.1f}" y="{y0 + 4:.1f}" text-anchor="middle" fill="#fff" '
                     f'font-weight="700">{step.number}</text>')
    return parts


def plan_svg(final: Scene, steps: list[Step], access: AccessReport | None) -> str:
    """Draws the room from above after the moves.

    Args:
        final: The scene after every step.
        steps: The steps, to draw their starts, ends and arrows.
        access: Reachable floor to shade, if known.

    Returns:
        An ``<svg>`` element.
    """
    frame = _frame(final, steps)
    parts = [f'<svg viewBox="0 0 {_WIDTH_PX:.0f} {frame.height:.0f}" '
             'xmlns="http://www.w3.org/2000/svg" font-family="system-ui" font-size="11">',
             '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
             'markerHeight="7" orient="auto-start-reverse"><path d="M0,0L10,5L0,10z" '
             'fill="#c2412d"/></marker></defs>']
    if final.floor is not None:
        parts.append(_cell_runs(final, final.floor.floor, frame, "#e7ebe9", 1.0))
        if access is not None:
            parts.append(_cell_runs(final, access.reachable, frame, "#2f9e6b", 0.25))
    parts += _rooms_svg(final, frame)
    parts += _objects_svg(final, frame, {s.object_id for s in steps})
    parts += _steps_svg(steps, frame)
    parts.append("</svg>")
    return "".join(parts)


def _access_text(final: Scene) -> tuple[str, bool]:
    """Summarizes who can still get around the room after the moves.

    Args:
        final: The scene after the moves.

    Returns:
        The sentence, and whether every object stays reachable.
    """
    wheelchair = check_access(final, WHEELCHAIR_WIDTH)
    if wheelchair is None:
        return "The floor is unknown: passages were not checked.", True
    if wheelchair.to_dict()["accessible"]:
        return (f"A wheelchair ({100 * WHEELCHAIR_WIDTH:.0f} cm) can still reach every object "
                "from the doors.", True)
    walking = check_access(final, WALKING_WIDTH)
    problems = [f"the {wheelchair.labels[i]} (#{i})" for i in wheelchair.unreachable]
    problems += [f"the {wheelchair.labels[i]} (#{i}) is blocked" for i in wheelchair.blocked_doors]
    walk_ok = walking is not None and walking.to_dict()["accessible"]
    return (f"A wheelchair ({100 * WHEELCHAIR_WIDTH:.0f} cm) cannot reach "
            f"{', '.join(problems)}. "
            + ("A person walking still can." if walk_ok else
               "A person walking cannot either."), False)


def instructions_page(final: Scene, steps: list[Step], title: str = "PIVOT",
                      save_pdf: bool = False) -> str:
    """Builds the printable instructions page.

    Args:
        final: The scene after every step.
        steps: The steps, in order.
        title: Name of the scene.
        save_pdf: Whether the page opens the browser's print dialog as soon as
            it is loaded, where "Save as PDF" gives the PDF.

    Returns:
        A standalone HTML page.
    """
    items = "".join(f"<li>{html.escape(step.text())}</li>" for step in steps)
    access_text, accessible = _access_text(final)
    two = sum(step.people > 1 for step in steps)
    summary = (f"{len(steps)} move{'s' * (len(steps) != 1)}"
               + (f", {two} of them with two people" if two else "") + ".")
    body = (f"<ol>{items}</ol>" if steps else
            "<p>No move yet: ask PIVOT to rearrange the room, then open this page again.</p>")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Moving plan - {html.escape(title)}</title><style>{_STYLE}</style></head>"
            f"<body{SAVE_PDF_ON_LOAD if save_pdf else ''}>"
            f"<h1>Moving plan</h1><div class='muted'>{html.escape(title)} · {summary} "
            f"Dashed: where an object starts; orange: where it ends; green: floor a "
            f"wheelchair can reach.</div>"
            f"<button onclick='print()'>Save as PDF or print</button>"
            f"{plan_svg(final, steps, check_access(final, WHEELCHAIR_WIDTH))}"
            f"<p class='access{'' if accessible else ' no'}'>{html.escape(access_text)}</p>"
            f"{body}</body></html>")
