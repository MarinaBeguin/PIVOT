"""Printable moving instructions: a plan seen from above and the numbered steps.

The plan shows the floor, the objects where they end up, each moved object's
start (dashed) and end (filled) joined by the numbered path it takes (the one
the 3D view plays), and the floor a
wheelchair can still reach from the doors after the moves.
"""

from __future__ import annotations

import html
import json
from importlib import resources
from typing import Any

import numpy as np

from pivot.geometry.types import BoolArray, FloatArray
from pivot.planning.access import WALKING_WIDTH, WHEELCHAIR_WIDTH, AccessReport, check_access
from pivot.planning.instructions import Step
from pivot.scene.graph import Scene, SceneObject, is_opening
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
                      f'font-weight="600" fill="#5b6661" class="room">'
                      f'{html.escape(room.name)}</text>')
    return labels


def _objects_svg(final: Scene, frame: _Frame, moved: set[int],
                 added: frozenset[int] = frozenset()) -> list[str]:
    """Draws the objects where they end up, moved and new ones highlighted, numbered.

    Args:
        final: The scene after every step.
        frame: Coordinate mapping.
        moved: Ids of the moved objects.
        added: Ids of the objects brought into the room.

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
        fill = "#5bbf8a" if obj.object_id in added else \
            "#f2b84b" if obj.object_id in moved else "#aab4b0"
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
        # The path the object takes, as the viewer plays it; a straight arrow without one.
        centers = [step.start.center[:2], *step.path[1:-1], step.end.center[:2]]
        points = [frame.point(np.asarray(c, dtype=float)) for c in centers]
        x0, y0 = points[0]
        line = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(f'<polyline points="{line}" fill="none" stroke="#c2412d" '
                     'stroke-width="2" stroke-linejoin="round" marker-end="url(#arrow)"/>')
        parts.append(f'<circle cx="{x0:.1f}" cy="{y0:.1f}" r="9" fill="#c2412d"/>'
                     f'<text x="{x0:.1f}" y="{y0 + 4:.1f}" text-anchor="middle" fill="#fff" '
                     f'font-weight="700">{step.number}</text>')
    return parts


def plan_svg(final: Scene, steps: list[Step], access: AccessReport | None,
             added: frozenset[int] = frozenset()) -> str:
    """Draws the room from above after the moves.

    Args:
        final: The scene after every step.
        steps: The steps, to draw their starts, ends and arrows.
        access: Reachable floor to shade, if known.
        added: Ids of the objects brought into the room, drawn apart.

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
    parts += _objects_svg(final, frame, {s.object_id for s in steps}, added)
    parts += _steps_svg(steps, frame)
    parts.append("</svg>")
    return "".join(parts)


def _access_data(final: Scene, moved: set[int] | None = None) -> dict[str, Any]:
    """Finds who can still get around the room after the moves.

    Args:
        final: The scene after the moves.
        moved: Ids of the moved objects, whose place in front of a door is noted.

    Returns:
        ``known`` (the floor is), ``accessible`` (a wheelchair reaches every
        object), ``unreachable`` and ``blocked_doors`` (``[label, id]``),
        ``walk_ok`` (a person walking reaches every object) and ``in_front``
        (``[label, id, door label, door id]`` for moved objects before a door).
    """
    wheelchair = check_access(final, WHEELCHAIR_WIDTH)
    if wheelchair is None:
        return {"known": False}
    in_front = [[wheelchair.labels[i], i, wheelchair.labels[door], door]
                for door, ids in wheelchair.obstructed_doors.items() for i in ids
                if i in (moved or set())]
    accessible = bool(wheelchair.to_dict()["accessible"])
    walking = None if accessible else check_access(final, WALKING_WIDTH)
    return {"known": True, "accessible": accessible,
            "unreachable": [[wheelchair.labels[i], i] for i in wheelchair.unreachable],
            "blocked_doors": [[wheelchair.labels[i], i] for i in wheelchair.blocked_doors],
            "walk_ok": accessible or (walking is not None and walking.to_dict()["accessible"]),
            "in_front": in_front}


def _access_text(access: dict[str, Any]) -> tuple[str, bool]:
    """Words who can still get around the room, in English.

    Args:
        access: See :func:`_access_data`.

    Returns:
        The sentence, and whether every object stays reachable and no moved
        object stands in front of a door.
    """
    if not access["known"]:
        return "The floor is unknown: passages were not checked.", True
    parts = [f"the {label} (#{i}) in front of the {door} (#{door_id})"
             for label, i, door, door_id in access["in_front"]]
    doors = f" Note: {', '.join(parts)}." if parts else ""
    if access["accessible"]:
        return (f"A wheelchair ({100 * WHEELCHAIR_WIDTH:.0f} cm) can still reach every object "
                "from the doors." + doors, not doors)
    problems = [f"the {label} (#{i})" for label, i in access["unreachable"]]
    problems += [f"the {label} (#{i}) is blocked" for label, i in access["blocked_doors"]]
    return (f"A wheelchair ({100 * WHEELCHAIR_WIDTH:.0f} cm) cannot reach "
            f"{', '.join(problems)}. "
            + ("A person walking still can." if access["walk_ok"] else
               "A person walking cannot either.") + doors, False)


# Words the page in the visitor's language, from the data written next to it: the
# English text is the server's, every other language is rewritten here, with the
# texts and object names of the viewer (i18n.js).
_TRANSLATE = """
(() => {
  const I = window.PivotI18n, d = JSON.parse(document.getElementById("plan-data").textContent);
  const lang = I.LANGUAGES[d.lang] ? d.lang : "en";
  if (lang === "en") return;
  document.documentElement.lang = lang;
  const t = (key, values) => I.text(lang, key, values);
  const name = (label, id) => t("planName", { label: I.name(lang, label), id });
  const length = m => m < 1 ? `${Math.round(100 * m)} cm`
    : `${m.toLocaleString(lang, { maximumFractionDigits: 1, minimumFractionDigits: 1 })} m`;
  const set = (id, text) => { const node = document.getElementById(id); if (node) node.textContent = text; };
  set("plan-title", t("planHeading"));
  document.querySelectorAll("text.room").forEach(node => { node.textContent = I.name(lang, node.textContent); });
  document.title = `${t("planHeading")} - ${d.title}`;
  const s = d.summary;
  set("plan-summary", `${d.title} · ` + (s.moves === 1 ? t("planMoveOne") : t("planMoves", { n: s.moves }))
    + (s.two ? t("planTwoPeople", { n: s.two }) : "") + (s.added ? t("planBroughtIn", { n: s.added }) : "")
    + (s.removed ? t("planTakenOut", { n: s.removed }) : "") + ". " + t("planLegend"));
  set("plan-save", t("planSave"));
  const a = d.access;
  let access = t("planAccessUnknown");
  if (a.known) {
    const note = a.in_front.length ? " " + t("planDoorNote", { items: a.in_front.map(([l, i, dl, di]) =>
      t("planInFront", { name: name(l, i), door: name(dl, di) })).join(", ") }) : "";
    const problems = a.unreachable.map(([l, i]) => name(l, i))
      .concat(a.blocked_doors.map(([l, i]) => t("planBlocked", { name: name(l, i) })));
    access = a.accessible ? t("planAccessOk", { cm: d.wheelchair_cm }) + note
      : t("planAccessNo", { cm: d.wheelchair_cm, names: problems.join(", ") }) + " "
        + t(a.walk_ok ? "planWalkOk" : "planWalkNo") + note;
  }
  set("plan-access", access);
  if (d.removed.length) set("plan-removed", t("planFirstOut", { names: d.removed.map(([l, i]) => name(l, i)).join(", ") }));
  if (d.added.length) set("plan-added", t("planThenIn", { names: d.added.map(([l, i, size]) =>
    `${name(l, i)} (${size} cm)`).join(", ") }));
  set("plan-empty", t("planNoMove"));
  document.querySelectorAll("#plan-steps li").forEach((item, k) => {
    const step = d.steps[k], who = name(step.label, step.id), far = length(step.distance_m);
    let text = t(step.rolled ? "stepRoll" : step.carried ? "stepCarry" : "stepSlide", { name: who, dist: far });
    if (step.rooms) {
      const doors = step.doors.map(door => door.id === null ? t("stepDoorway")
        : door.spare_cm === null ? name("door", door.id) : t("stepDoor", { door: name("door", door.id), cm: door.spare_cm }));
      text += t("stepRooms", { from: I.name(lang, step.rooms.from), to: I.name(lang, step.rooms.to), doors: doors.join(", ") });
    }
    if (step.landmark) {
      const near = name(step.landmark.label, step.landmark.id);
      text += " " + (step.landmark.gap_m < 0.03 ? t("stepAgainst", { name: near })
        : t("stepNear", { dist: length(step.landmark.gap_m), name: near }));
    }
    if (step.turn_deg) text += t(step.turn_deg > 0 ? "stepTurnLeft" : "stepTurnRight", { deg: Math.abs(step.turn_deg) });
    text += ".";
    if (step.rolled) text += " " + t("stepUnroll");
    if (step.people > 1) text += " " + t("stepTwo");
    item.textContent = text;
  });
})();
"""


# The viewer's texts, which translate the page in the visitor's language.
_LANGUAGES = resources.files("pivot.viz").joinpath("i18n.js").read_text(encoding="utf-8")

def _summary(steps: list[Step], removed: list[tuple[str, int]],
             added: list[SceneObject]) -> str:
    """Counts the moves, in English.

    Args:
        steps: The steps, in order.
        removed: Objects taken out of the room.
        added: Objects brought into the room.

    Returns:
        A sentence such as "3 moves, 1 of them with two people."
    """
    two = sum(step.people > 1 for step in steps)
    return (f"{len(steps)} move{'s' * (len(steps) != 1)}"
            + (f", {two} of them with two people" if two else "")
            + (f", {len(added)} brought in" if added else "")
            + (f", {len(removed)} taken out" if removed else "") + ".")


def _plan_body(steps: list[Step], removed: list[tuple[str, int]], added: list[SceneObject],
               sizes: dict[int, str]) -> str:
    """Lists, in English, what to take out, the steps, then what to bring in.

    Args:
        steps: The steps, in order.
        removed: Objects taken out of the room, as ``(label, id)``.
        added: Objects brought into the room.
        sizes: Size of each added object, in centimeters, by id.

    Returns:
        The HTML of the list.
    """
    items = "".join(f"<li>{html.escape(step.text())}</li>" for step in steps)
    body = (f"<ol id='plan-steps'>{items}</ol>" if steps else
            "" if added or removed else
            "<p id='plan-empty'>No move yet: ask PIVOT to rearrange the room, then open this "
            "page again.</p>")
    if removed:
        names = ", ".join(f"the {label} (#{object_id})" for label, object_id in removed)
        body = (f"<p class='removed' id='plan-removed'>First take out of the room: "
                f"{html.escape(names)}.</p>" + body)
    if added:
        names = ", ".join(f"the {o.label} (#{o.object_id}), {sizes[o.object_id]} cm"
                          for o in added)
        body += f"<p class='added' id='plan-added'>Then bring in: {html.escape(names)}.</p>"
    return body


def instructions_page(final: Scene, steps: list[Step], title: str = "PIVOT",
                      save_pdf: bool = False,
                      removed: list[tuple[str, int]] | None = None,
                      added: list[SceneObject] | None = None, language: str = "en") -> str:
    """Builds the printable instructions page, in the visitor's language.

    The page is written in English, with its facts as data; in another
    language, a script rewrites it with the viewer's texts (see i18n.js).

    Args:
        final: The scene after every step.
        steps: The steps, in order.
        title: Name of the scene.
        save_pdf: Whether the page opens the browser's print dialog as soon as
            it is loaded, where "Save as PDF" gives the PDF.
        removed: Objects taken out of the room, as ``(label, id)``: they go first.
        added: Objects brought into the room: they come after, where the plan shows them.
        language: Code of the visitor's language.

    Returns:
        A standalone HTML page.
    """
    added, removed = added or [], removed or []
    access = _access_data(final, {step.object_id for step in steps})
    access_text, accessible = _access_text(access)
    sizes = {o.object_id: " x ".join(f"{100 * v:.0f}" for v in o.box.size) for o in added}
    body = _plan_body(steps, removed, added, sizes)
    data = {"lang": language, "title": title, "wheelchair_cm": round(100 * WHEELCHAIR_WIDTH),
            "summary": {"moves": len(steps), "two": sum(step.people > 1 for step in steps),
                        "added": len(added),
                        "removed": len(removed)},
            "access": access, "steps": [step.to_dict() for step in steps],
            "removed": [[label, i] for label, i in removed],
            "added": [[o.label, o.object_id, sizes[o.object_id]] for o in added]}
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>Moving plan - {html.escape(title)}</title><style>{_STYLE}</style></head>"
            f"<body{SAVE_PDF_ON_LOAD if save_pdf else ''}>"
            f"<h1 id='plan-title'>Moving plan</h1><div class='muted' id='plan-summary'>"
            f"{html.escape(title)} · {_summary(steps, removed, added)} "
            f"Dashed: where an object starts; orange: where it ends; green: a new "
            f"object; shaded green: floor a wheelchair can reach.</div>"
            f"<button id='plan-save' onclick='print()'>Save as PDF or print</button>"
            f"{plan_svg(final, steps, check_access(final, WHEELCHAIR_WIDTH),
                        frozenset(o.object_id for o in added))}"
            f"<p class='access{'' if accessible else ' no'}' id='plan-access'>"
            f"{html.escape(access_text)}</p>"
            f"{body}"
            f"<script type='application/json' id='plan-data'>"
            f"{json.dumps(data).replace('</', '<' + chr(92) + '/')}</script>"
            f"<script>{_LANGUAGES}</script><script>{_TRANSLATE}</script></body></html>")
