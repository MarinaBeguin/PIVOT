"""HTTP server for the interactive viewer.

The page is the viewer of :mod:`pivot.viz.web` with its controls enabled. It
calls the same tools as the agent (:mod:`pivot.agent.tools`) on the visitor's
own session, so a move made from the page is verified exactly like a move made
by the agent, and both can be undone. With ``--chat``, the page also takes
requests in words, answered by the Muse agent on that same session (the API
key is read from ``--env-file``), typed or spoken (:mod:`pivot.server.speech`).
With ``--public``, requests are priced and limited (:mod:`pivot.server.limits`).

Run locally::

    python -m pivot.server --scene-dir outputs/b20a261fdf --port 8000 [--chat]
    python -m pivot.server --scene-dir Office=outputs/office --scene-dir Classroom=outputs/class
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import os
import re
import tempfile
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from pivot.agent.loop import Agent
from pivot.agent.tools import TOOLS_BY_NAME, Session, call_tool
from pivot.geometry.types import FloatArray
from pivot.meta.client import MetaConfig, Usage
from pivot.meta.image import photo_view
from pivot.meta.photo import PhotoObject, object_from_photo
from pivot.meta.sam import SamClient
from pivot.planning.instructions import instructions
from pivot.planning.motion import MotionPlan
from pivot.scene.graph import Scene, carried_objects
from pivot.server.guard import Guard, guard_from_env
from pivot.server.limits import (
    IMAGE_PRICE, PHOTO_PRICE, TRANSCRIBE_PER_SECOND, Budget, Meter, SpendLedger, chat_cost,
)
from pivot.server.speech import Transcriber, read_wav, transcriber_from_env
from pivot.server.visitors import COOKIE, Visitor, Visitors
from pivot.viz.plan import instructions_page
from pivot.viz.pointcloud import ColorArray
from pivot.viz.web import Features, RoomMenu, render_viewer

_MUTATING = {"move_object", "make_room_and_move", "undo", "remove_object", "add_object",
             "resize_object"}
# About a minute of 16 kHz 16-bit mono audio.
MAX_AUDIO_BYTES = 2_000_000
# A 3D view sent for a realistic one: a PNG of the page, at most this big.
MAX_IMAGE_BYTES = 12_000_000
# Edits a 3D view (PNG) into a realistic one, given photos of objects brought in.
Painter = Callable[[bytes, list[tuple[str, bytes]]], bytes]
# Names the object in a photo, sizes it and cuts it out; None if there is none.
Photographer = Callable[[bytes], PhotoObject | None]


def _objects(session: Session) -> list[dict[str, Any]]:
    """Serializes the current objects of a session for the page.

    Args:
        session: The session.

    Returns:
        Objects sorted by id, as in the viewer payload.
    """
    return [o.to_dict() for o in sorted(session.scene.objects.values(), key=lambda o: o.object_id)]


def _motions(scene: Scene, motions: list[MotionPlan]) -> list[dict[str, Any]]:
    """Serializes executed motions for the page to animate.

    Args:
        scene: The scene before the first motion.
        motions: The motions, in order.

    Returns:
        Each motion with the objects it carried along.
    """
    result = []
    for motion in motions:
        result.append({**motion.to_dict(),
                       "carried_along": carried_objects(scene, motion.object_id)})
        scene = scene.moved(motion.object_id, motion.waypoints[-1])
    return result


def _chat_reply(visitor: Visitor, agent_factory: Callable[[Session], Agent], message: str,
                meter: Meter | None, key: str, language: str | None = None) -> JSONResponse:
    """Answers a request in words on a visitor's session, and charges it.

    Args:
        visitor: The visitor.
        agent_factory: Creates the visitor's agent on first use.
        message: The request.
        meter: Prices the request, if the deployment has a budget.
        key: The visitor's key in the ledger.
        language: Code of the language of the visitor's page.

    Returns:
        The answer, the tools called, the motions executed (to animate) and
        the updated objects; or an error.
    """
    if meter is not None and (reason := meter.ledger.refusal(key)):
        return JSONResponse({"error": reason}, status_code=429)
    session = visitor.session
    agent: Agent = visitor.agent or agent_factory(session)
    visitor.agent = agent
    before, count = session.scene, len(session.motions)
    spent = Usage(agent.usage.calls, agent.usage.prompt_tokens, agent.usage.completion_tokens)
    try:
        run = agent.ask(message, language)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # An API failure must reach the page, not break the server.
        return JSONResponse({"error": f"The agent could not answer: {error}"})
    finally:
        if meter is not None:
            meter.ledger.record(key, meter.chat_price(Usage(
                agent.usage.calls - spent.calls,
                agent.usage.prompt_tokens - spent.prompt_tokens,
                agent.usage.completion_tokens - spent.completion_tokens)))
    return JSONResponse({
        "answer": run.answer, "finished": run.finished,
        "steps": [{"tool": s.tool, "arguments": s.arguments} for s in run.steps],
        "motions": _motions(before, session.motions[count:]),
        "objects": _objects(session)})


def _decode_data_url(url: str) -> bytes | None:
    """Reads the bytes of a ``data:...;base64,...`` address.

    Args:
        url: The address.

    Returns:
        The bytes, or None if it is not a base64 data address.
    """
    head, _, data = url.partition(",")
    if not head.startswith("data:") or not head.endswith(";base64"):
        return None
    try:
        return base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError):
        return None


def _render_reply(visitor: Visitor, painter: Painter, view_url: str, meter: Meter | None,
                  key: str) -> JSONResponse:
    """Draws a realistic view of a visitor's room, and charges it.

    Args:
        visitor: The visitor; the photos of the objects they brought in are drawn.
        painter: Edits the 3D view into a realistic one.
        view_url: The 3D view, as a PNG data address.
        meter: Prices the request, if the deployment has a budget.
        key: The visitor's key in the ledger.

    Returns:
        ``{"image": "data:image/...;base64,..."}``, or an error.
    """
    if meter is not None and (reason := meter.ledger.refusal(key)):
        return JSONResponse({"error": reason}, status_code=429)
    view = _decode_data_url(view_url)
    if view is None or len(view) > MAX_IMAGE_BYTES:
        return JSONResponse({"error": "The view could not be read."}, status_code=400)
    brought = [photo for object_id, photo in visitor.photos.items()
               if object_id in visitor.session.scene.objects]
    try:
        image = painter(view, brought)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # A failed drawing must reach the page, not break the server.
        return JSONResponse({"error": f"Could not draw the view: {error}"})
    finally:
        if meter is not None:
            meter.ledger.record(key, meter.image_price)
    kind = "png" if image.startswith(b"\x89PNG") else "jpeg"
    return JSONResponse({"image": f"data:image/{kind};base64,{base64.b64encode(image).decode()}"})


def _photo_reply(visitor: Visitor, photographer: Photographer, body: dict[str, Any],
                 meter: Meter | None, key: str) -> JSONResponse:
    """Brings an object into a visitor's room from a photo, and charges it.

    Args:
        visitor: The visitor.
        photographer: Names, sizes and cuts out the object of a photo.
        body: The photo as a data address, and optionally where the object should go.
        meter: Prices the request, if the deployment has a budget.
        key: The visitor's key in the ledger.

    Returns:
        The object added, its look, its picture and the updated objects; or
        why it was not added.
    """
    if meter is not None and (reason := meter.ledger.refusal(key)):
        return JSONResponse({"error": reason}, status_code=429)
    picture = _decode_data_url(str(body.get("image", "")))
    if picture is None or len(picture) > MAX_IMAGE_BYTES:
        return JSONResponse({"error": "The photo could not be read."}, status_code=400)
    try:
        seen = photographer(picture)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # A failed model call must reach the page, not break the server.
        return JSONResponse({"error": f"Could not look at the photo: {error}"})
    finally:
        if meter is not None:
            meter.ledger.record(key, meter.photo_price)
    if seen is None:
        return JSONResponse({"added": False, "reason": "no_object"})
    where = {k: float(body[k]) for k in ("x", "y") if body.get(k) is not None}
    result: dict[str, Any] = json.loads(call_tool(visitor.session, "add_object", json.dumps(
        {"label": seen.label, **dict(zip(("width_m", "depth_m", "height_m"), seen.size)),
         **(where if len(where) == 2 else {})})))
    if result.get("added"):
        object_id = int(result["object_id"])
        visitor.photos[object_id] = (f"the new {seen.label} (#{object_id})", seen.picture)
    cutout = "data:image/png;base64," + base64.b64encode(seen.picture).decode()
    return JSONResponse({**result, "look": seen.look, "picture": cutout,
                         "objects": _objects(visitor.session)})


def _audio_seconds(audio: bytes) -> float:
    """Measures a recording, to price its transcription.

    Args:
        audio: WAV bytes.

    Returns:
        Its duration in seconds; for an unreadable file, as if it were 16 kHz
        16-bit mono.
    """
    try:
        samples, rate = read_wav(audio)
        return len(samples) / rate
    except (ValueError, EOFError, wave.Error):
        return len(audio) / 32000


# The routes are local functions: each counts as a local variable.
def _render_pages(rooms: list[Room], features: Features) -> dict[str, str]:
    """Renders the viewer page of every room, with the menu of all rooms.

    Args:
        rooms: The rooms.
        features: What the server offers.

    Returns:
        The pages, by room id.
    """
    menu = [(room.room_id, room.title) for room in rooms]
    return {room.room_id: render_viewer(room.scene, room.points, room.colors, room.title,
                                        features, RoomMenu(room.room_id, menu))
            for room in rooms}


# pylint: disable-next=too-many-locals
def create_app(rooms: list[Room],
               agent_factory: Callable[[Session], Agent] | None = None,
               transcriber: Transcriber | None = None,
               meter: Meter | None = None,
               painter: Painter | None = None,
               photographer: Photographer | None = None,
               guard: Guard | None = None) -> FastAPI:
    """Creates the web application for one or several rooms.

    The room of a request is given by its ``scene`` query parameter (the first
    room by default); each visitor works on their own copy of each room (see
    :mod:`pivot.server.visitors`).

    Args:
        rooms: The rooms, the default first; ids must be unique.
        agent_factory: Creates the agent of a session; None disables requests in words.
        transcriber: Turns spoken requests into text; None disables the microphone.
        meter: Prices requests and enforces a budget; None for a private server.
        painter: Edits a 3D view into a realistic one; None disables realistic views.
        photographer: Names an object in a photo, sizes it and cuts it out;
            None disables objects from photos.
        guard: Screens requests in words before the agent sees them (Llama
            Prompt Guard, Llama Guard); None screens nothing.

    Returns:
        The application.
    """
    app = FastAPI(title="PIVOT")
    # The page carries the point cloud: several megabytes, which compress well.
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    by_id = {room.room_id: room for room in rooms}
    visitors = Visitors({room.room_id: room.scene for room in rooms})
    pages = _render_pages(rooms, Features(
        interactive=True, chat=agent_factory is not None,
        voice=agent_factory is not None and transcriber is not None,
        render=painter is not None, photos=photographer is not None))

    @app.middleware("http")
    async def visitor_session(request: Request,
                              call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Attaches the visitor's session to the request, and sets their cookie.

        Args:
            request: The request.
            call_next: The route.

        Returns:
            The route's response.
        """
        room = by_id.get(request.query_params.get("scene", ""), rooms[0])
        token, request.state.visitor = visitors.get(request.cookies.get(COOKIE), room.room_id)
        request.state.token, request.state.room = token, room
        response = await call_next(request)
        if request.cookies.get(COOKIE) != token:
            response.set_cookie(COOKIE, token, httponly=True, samesite="lax")
        return response

    def key(request: Request) -> str:
        """Identifies the visitor of a request in the ledger.

        Args:
            request: The request.

        Returns:
            The visitor key; empty without a meter.
        """
        if meter is None:
            return ""
        return meter.visitor(request.headers, request.client.host if request.client else None)

    @app.get("/", response_class=HTMLResponse)
    def viewer(request: Request) -> str:
        """Serves the viewer page of the requested room.

        Args:
            request: The request.

        Returns:
            The HTML page.
        """
        return pages[request.state.room.room_id]

    @app.get("/instructions", response_class=HTMLResponse)
    def moving_plan(request: Request) -> str:
        """Serves the moving instructions of the visitor's moves so far.

        With ``?pdf=1`` the page opens the browser's dialog to save it as a PDF.

        Args:
            request: The request.

        Returns:
            A printable page: plan from above and numbered steps.
        """
        session: Session = request.state.visitor.session
        initial = session.history[0] if session.history else session.scene
        removed = [(o.label, o.object_id) for o in initial.objects.values()
                   if o.object_id not in session.scene.objects]
        added = [o for o in session.scene.objects.values() if o.object_id not in initial.objects]
        return instructions_page(session.scene, instructions(initial, session.motions),
                                 request.state.room.title,
                                 save_pdf=request.query_params.get("pdf") == "1",
                                 removed=removed, added=added)

    @app.get("/api/objects")
    def objects(request: Request) -> list[dict[str, Any]]:
        """Lists the visitor's current objects.

        Args:
            request: The request.

        Returns:
            The objects, sorted by id.
        """
        return _objects(request.state.visitor.session)

    @app.post("/api/tools/{name}")
    def tool(name: str, arguments: dict[str, Any], request: Request) -> JSONResponse:
        """Calls a geometric tool on the visitor's session.

        Args:
            name: Tool name, see :data:`pivot.agent.tools.TOOLS`.
            arguments: Tool arguments.
            request: The request.

        Returns:
            The tool result; moves and undos also carry the updated objects.
        """
        if name not in TOOLS_BY_NAME:
            return JSONResponse({"error": f"Unknown tool {name!r}."}, status_code=404)
        session = request.state.visitor.session
        result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
        if name in _MUTATING:
            result["objects"] = _objects(session)
        return JSONResponse(result)

    @app.post("/api/chat")
    def chat(body: dict[str, str], request: Request) -> JSONResponse:
        """Answers a request in words with the agent, on the visitor's session.

        Args:
            body: ``{"message": ..., "language": ...}``, the language of the page
                being optional.
            request: The request.

        Returns:
            The answer, the tools called, the motions executed (to animate)
            and the updated objects; or why the request is refused.
        """
        if agent_factory is None:
            return JSONResponse({"error": "Requests in words are not enabled."},
                                status_code=404)
        if guard is not None and (reason := guard(body.get("message", ""))):
            # Refused before any API call: it costs nothing.
            return JSONResponse({"error": reason, "screened": True})
        return _chat_reply(request.state.visitor, agent_factory, body.get("message", ""),
                           meter, key(request), body.get("language"))

    @app.post("/api/transcribe")
    async def transcribe(request: Request) -> JSONResponse:
        """Turns a spoken request into text.

        Args:
            request: Its body is a WAV recording.

        Returns:
            ``{"text": ...}``, or an error.
        """
        if transcriber is None:
            return JSONResponse({"error": "Spoken requests are not enabled."}, status_code=404)
        visitor_key = key(request)
        if meter is not None and (reason := meter.ledger.refusal(visitor_key)):
            return JSONResponse({"error": reason}, status_code=429)
        audio = await request.body()
        if len(audio) > MAX_AUDIO_BYTES:
            return JSONResponse({"error": "Please keep spoken requests under a minute."},
                                status_code=413)
        labels = sorted({o.label for o in request.state.visitor.session.scene.objects.values()})
        try:
            text = transcriber(audio, labels)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # A transcription failure must reach the page, not break the server.
            return JSONResponse({"error": f"Could not transcribe: {error}"})
        finally:
            if meter is not None:
                meter.ledger.record(visitor_key, _audio_seconds(audio) * meter.audio_per_second,
                                    request=False)
        return JSONResponse({"text": text})

    @app.post("/api/render")
    def render(body: dict[str, str], request: Request) -> JSONResponse:
        """Draws a realistic view of the room from the visitor's 3D view.

        Args:
            body: ``{"image": "data:image/png;base64,..."}``, the 3D view.
            request: The request.

        Returns:
            ``{"image": "data:image/...;base64,..."}``, or an error.
        """
        if painter is None:
            return JSONResponse({"error": "Realistic views are not enabled."}, status_code=404)
        return _render_reply(request.state.visitor, painter, body.get("image", ""), meter,
                             key(request))

    @app.post("/api/photo")
    def photo(body: dict[str, Any], request: Request) -> JSONResponse:
        """Brings an object into the room from a photo of it.

        Args:
            body: ``{"image": "data:image/...;base64,...", "x": ..., "y": ...}``;
                the point where it should go is optional.
            request: The request.

        Returns:
            The object added (as ``add_object`` says), its look, its picture,
            and the updated objects; or why it was not added.
        """
        if photographer is None:
            return JSONResponse({"error": "Objects from photos are not enabled."},
                                status_code=404)
        return _photo_reply(request.state.visitor, photographer, body, meter, key(request))

    @app.get("/api/limits")
    def limits(request: Request) -> dict[str, Any]:
        """Says what the visitor may still use today on a public server.

        Args:
            request: The request.

        Returns:
            ``{"public": false}`` for a private server; else the requests left.
        """
        if meter is None:
            return {"public": False}
        return {"public": True, **meter.ledger.remaining(key(request))}

    @app.post("/api/reset")
    def reset(request: Request) -> list[dict[str, Any]]:
        """Restores the visitor's initial scene and starts a new conversation.

        Args:
            request: The request.

        Returns:
            The objects, sorted by id.
        """
        return _objects(visitors.reset(request.state.token, request.state.room.room_id).session)

    return app


@dataclass(frozen=True)
class Room:
    """A room the server offers: its scene, the point cloud shown, and its name.

    Attributes:
        room_id: Short identifier, used in addresses (``?scene=office``).
        title: Name shown to visitors.
        scene: The scene every visitor starts from.
        points: (N, 3) point cloud shown in the viewer.
        colors: (N, 3) RGB colors.
    """

    room_id: str
    title: str
    scene: Scene
    points: FloatArray
    colors: ColorArray


def _subsample(points: FloatArray, colors: ColorArray,
               max_points: int | None) -> tuple[FloatArray, ColorArray]:
    """Keeps at most ``max_points`` points of the cloud, picked at random (reproducibly).

    Args:
        points: (N, 3) points.
        colors: (N, 3) colors.
        max_points: Most points kept; all if None.

    Returns:
        The points and colors kept.
    """
    if max_points is None or len(points) <= max_points:
        return points, colors
    keep = np.sort(np.random.default_rng(0).choice(len(points), max_points, replace=False))
    return points[keep], colors[keep]


def load_room(spec: str, max_points: int | None = None) -> Room:
    """Loads a room from a directory written by :mod:`pivot.build`.

    Args:
        spec: ``Title=path`` or a bare path; the title defaults to the
            directory's name, and the room id is the title in lowercase.
        max_points: Most points of the cloud sent to the page; all if None.

    Returns:
        The room.
    """
    title, _, path = spec.rpartition("=")
    scene_dir = Path(path)
    title = title or scene_dir.name
    cloud = np.load(scene_dir / "cloud.npz")
    points, colors = _subsample(cloud["points"], cloud["colors"], max_points)
    room_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return Room(room_id, title, Scene.load(scene_dir / "scene.json"), points, colors)


def _public_meter(config: MetaConfig | None) -> Meter:
    """Prices and limits the requests of a public server, from the environment.

    The budget comes from :meth:`Budget.from_env`; the ledger is kept at
    ``PIVOT_LEDGER`` (in memory if unset); the visitor key salt is
    ``PIVOT_VISITOR_SALT`` or derived from the API key; ``PIVOT_PROXY_HOPS``
    proxies are trusted. Open-weight models cost nothing.

    Args:
        config: API configuration, if requests in words are enabled.

    Returns:
        The meter.
    """
    api = config is not None and os.environ.get("PIVOT_BACKEND", "api").lower() != "open"
    ledger = os.environ.get("PIVOT_LEDGER")
    salt = os.environ.get("PIVOT_VISITOR_SALT") or hmac.new(
        (config.api_key if config else "pivot").encode(), b"pivot-visitors",
        hashlib.sha256).hexdigest()
    return Meter(SpendLedger(Budget.from_env(), Path(ledger) if ledger else None), salt,
                 int(os.environ.get("PIVOT_PROXY_HOPS", "1")),
                 partial(chat_cost, config.agent_model) if api and config else
                 (lambda usage: 0.0),
                 TRANSCRIBE_PER_SECOND if api else 0.0,
                 IMAGE_PRICE if api else 0.0,
                 PHOTO_PRICE if api else 0.0)


def load_app(scene_dirs: list[str], env_file: Path | None = None, max_points: int | None = None,
             public: bool = False) -> FastAPI:
    """Creates the application for rooms written by :mod:`pivot.build`.

    Args:
        scene_dirs: The rooms, as accepted by :func:`load_room`, the default first.
        env_file: API configuration file; requests in words are enabled if given.
        max_points: Most points of each cloud sent to the page; all if None.
        public: Whether the server is open to anyone: requests are then priced
            and limited (see :func:`_public_meter`).

    Returns:
        The application.
    """
    config = MetaConfig.from_env(env_file) if env_file else None
    # A conversation may rearrange a whole room: many more steps than one move.
    factory = partial(Agent, config=config, max_steps=40) if config else None
    api = config is not None and os.environ.get("PIVOT_BACKEND", "api").lower() != "open"
    return create_app([load_room(spec, max_points) for spec in scene_dirs],
                      agent_factory=factory,
                      transcriber=transcriber_from_env(config) if config else None,
                      meter=_public_meter(config) if public else None,
                      painter=partial(photo_view, config) if api and config else None,
                      photographer=_photographer(config) if api and config else None,
                      guard=guard_from_env() if config else None)


def _photographer(config: MetaConfig) -> Photographer:
    """Builds the function that brings objects in from photos, with the Meta API.

    Args:
        config: API configuration.

    Returns:
        Muse Spark names and sizes the object; SAM 3.1 cuts it out (its
        answers cached under ``PIVOT_CACHE``, or a temporary folder).
    """
    cache = Path(os.environ.get("PIVOT_CACHE", tempfile.gettempdir())) / "sam_photos"
    sam = SamClient(config, cache)
    return partial(object_from_photo, config, segment=sam.segment_image)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--scene-dir", action="append", required=True,
                        help="a room: Title=path or a path; repeat for several rooms")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chat", action="store_true", help="answer requests in words")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--max-points", type=int, default=None,
                        help="most points of the cloud sent to the page")
    parser.add_argument("--public", action="store_true",
                        help="open to anyone: price and limit requests (see load_app)")
    args = parser.parse_args(argv)

    # Imported here: only needed to serve, not to build or test the app.
    import uvicorn  # pylint: disable=import-outside-toplevel

    uvicorn.run(load_app(args.scene_dir, args.env_file if args.chat else None, args.max_points,
                         args.public),
                host=args.host, port=args.port)
