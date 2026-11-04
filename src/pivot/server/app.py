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
import gzip
import hashlib
import hmac
import json
import math
import os
import re
import tempfile
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from pivot.agent.loop import Agent
from pivot.agent.tools import TOOLS_BY_NAME, Session, call_tool
from pivot.geometry.box import OrientedBox
from pivot.geometry.types import FloatArray
from pivot.lifting.completion import prepared_scene
from pivot.planning.arrival import arrive, with_new_object
from pivot.meta.client import MetaConfig, Usage
from pivot.meta.photo import Described, PhotoObject, cut_object, describe_objects
from pivot.meta.sam import SamClient
from pivot.planning.instructions import instructions
from pivot.planning.motion import MotionPlan
from pivot.scene.graph import Scene, carried_objects
from pivot.server.guard import Guard, guard_from_env
from pivot.server.limits import (
    OWNER, PHOTO_PRICE, TRANSCRIBE_PER_SECOND, Budget, Meter, SpendLedger, chat_cost,
)
from pivot.server.speech import Transcriber, read_wav, transcriber_from_env
from pivot.server.uploads import VideoUploads
from pivot.server.cloud_uploads import CloudVideoUploads
from pivot.server.video_routes import (
    OWNER_COOKIE, add_video_routes, forget_video, prepare_video_request, remember_video_owner,
)
from pivot.server.visitors import COOKIE, Visitor, Visitors
from pivot.server.voice import MAX_TEXT, MMS_VOICES, Speaker, speaker_from_env
from pivot.viz.plan import instructions_page
from pivot.viz.pointcloud import ColorArray
from pivot.viz.web import Features, RoomMenu, render_viewer

_MUTATING = {"move_object", "make_room_and_move", "make_way", "undo", "remove_object",
             "add_object", "resize_object"}
# About a minute of 16 kHz 16-bit mono audio.
MAX_AUDIO_BYTES = 2_000_000
# A photo sent by a visitor, at most this big.
MAX_IMAGE_BYTES = 12_000_000


@dataclass(frozen=True)
class Photographer:
    """Brings objects in from photos.

    Attributes:
        describe: Names and sizes the objects of a photo, the most prominent first.
        cut: Cuts one of them out of the photo.
    """

    describe: Callable[[bytes], list[Described]]
    cut: Callable[[bytes, Described], PhotoObject]


# Marks the owner's browser, see Meter.owner_token.
# Points of a room built from a visitor's video, sent to the page.
_UPLOAD_POINTS = 200_000


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
    before, count, spent_before = session.scene, len(session.motions), session.spent
    spent = Usage(agent.usage.calls, agent.usage.prompt_tokens, agent.usage.completion_tokens)
    visitor.progress = []
    visitor.stop.clear()
    try:
        run = agent.ask(message, language, on_step=lambda tool, arguments: visitor.progress.append(
            {"tool": tool, "object_id": arguments.get("object_id")}),
            should_stop=visitor.stop.is_set)
    except Exception as error:  # pylint: disable=broad-exception-caught
        # An API failure must reach the page, not break the server.
        return JSONResponse({"error": f"The agent could not answer: {error}"})
    finally:
        if meter is not None:
            # Tokens of the agent, and what its tools spent (web searches).
            meter.ledger.record(key, meter.chat_price(Usage(
                agent.usage.calls - spent.calls,
                agent.usage.prompt_tokens - spent.prompt_tokens,
                agent.usage.completion_tokens - spent.completion_tokens))
                + session.spent - spent_before)
    return JSONResponse({
        "answer": run.answer, "finished": run.finished, "stopped": run.stopped,
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


def _ghost_box(box: OrientedBox) -> dict[str, Any]:
    """Describes a box for the page to draw an object's ghost there.

    Args:
        box: The box.

    Returns:
        Its center, yaw in degrees and size.
    """
    return {"x": round(float(box.center[0]), 3), "y": round(float(box.center[1]), 3),
            "z": round(float(box.center[2]), 3), "yaw_deg": round(math.degrees(box.yaw), 1),
            "size_m": [round(float(v), 3) for v in box.size]}


def _remember_owner(request: Request, response: Response, meter: Meter | None) -> None:
    """Marks the owner's browser, once they opened a page with ``?owner=<key>``.

    Args:
        request: The request.
        response: Its response, which gets the owner's cookie.
        meter: The server's meter; None for a private server.
    """
    if meter is not None and meter.owner_key and hmac.compare_digest(
            request.query_params.get("owner", ""), meter.owner_key):
        response.set_cookie(OWNER_COOKIE, meter.owner_token(), httponly=True,
                            samesite="lax", max_age=30 * 24 * 3600)


def _room_adder(rooms: list[Room], by_id: dict[str, Room], visitors: Visitors,
                pages: tuple[dict[str, str], dict[str, bytes]],
                features: Features) -> Callable[[str, Path], None]:
    """Makes the function that serves a room built from a visitor's video.

    Args:
        rooms: The server's rooms.
        by_id: The rooms by id, to which the new room is added.
        visitors: The visitors' copies of the rooms.
        pages: The viewer pages by room id, plain and compressed.
        features: What the pages offer.

    Returns:
        A function of the job's id and the scene directory it built.
    """
    plain, compressed = pages

    def add_room(room_id: str, scene_dir: Path) -> None:
        """Serves a room built from a visitor's video, under the job's id.

        Args:
            room_id: The job's id.
            scene_dir: The scene it built.
        """
        if room_id in by_id:
            return
        room = replace(load_room(f"Your video={scene_dir}", _UPLOAD_POINTS), room_id=room_id)
        by_id[room_id] = room
        visitors.add_room(room_id, room.scene)
        menu = RoomMenu(room_id, [*((r.room_id, r.title) for r in rooms), (room_id, room.title)])
        plain[room_id] = render_viewer(room.scene, room.points, room.colors, room.title,
                                       features, menu)
        compressed[room_id] = gzip.compress(plain[room_id].encode(), compresslevel=6)

    return add_room




def _chosen(body: dict[str, Any]) -> Described | None:
    """Reads the object the visitor picked among those of a photo, if any.

    Args:
        body: The request body; ``choice`` is ``{"label", "size_m", "look"}``.

    Returns:
        The object, or None if none was picked or it cannot be read.
    """
    choice = body.get("choice")
    if not isinstance(choice, dict):
        return None
    try:
        width, depth, height = (min(max(float(v), 0.05), 6.0) for v in choice["size_m"])
        return Described(str(choice["label"])[:60], (width, depth, height),
                         str(choice.get("look", ""))[:120])
    except (KeyError, TypeError, ValueError):
        return None


def _photo_reply(photographer: Photographer, body: dict[str, Any],
                 meter: Meter | None, key: str) -> JSONResponse:
    """Recognizes the objects of a photo, and charges it; the visitor then places one.

    When the photo shows several objects, they are listed for the visitor to
    pick one, which a second request (with ``choice``) cuts out. The object
    is not added here: the visitor chooses its spot, where it is added
    (``add_object`` at that spot, see :mod:`pivot.planning.arrival`).

    Args:
        photographer: Names, sizes and cuts out the objects of a photo.
        body: The photo as a data address, and the object picked, if any.
        meter: Prices the request, if the deployment has a budget.
        key: The visitor's key in the ledger.

    Returns:
        What the object is, its estimated size, its look and its picture; the
        objects to pick from; or why none was recognized.
    """
    if meter is not None and (reason := meter.ledger.refusal(key)):
        return JSONResponse({"error": reason}, status_code=429)
    picture = _decode_data_url(str(body.get("image", "")))
    if picture is None or len(picture) > MAX_IMAGE_BYTES:
        return JSONResponse({"error": "The photo could not be read."}, status_code=400)
    chosen = _chosen(body)
    try:
        if chosen is None:
            found = photographer.describe(picture)
            if meter is not None:
                meter.ledger.record(key, meter.photo_price)
            if len(found) > 1:
                return JSONResponse({"recognized": True, "choices": [
                    {"label": d.label, "size_m": [round(v, 2) for v in d.size], "look": d.look}
                    for d in found]})
            chosen = found[0] if found else None
        seen = photographer.cut(picture, chosen) if chosen is not None else None
    except Exception as error:  # pylint: disable=broad-exception-caught
        # A failed model call must reach the page, not break the server.
        return JSONResponse({"error": f"Could not look at the photo: {error}"})
    if seen is None:
        return JSONResponse({"recognized": False, "reason": "no_object"})
    cutout = "data:image/png;base64," + base64.b64encode(seen.picture).decode()
    return JSONResponse({"recognized": True, "label": seen.label,
                         "size_m": [round(v, 2) for v in seen.size], "look": seen.look,
                         "picture": cutout})


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


def _chat_side_routes(app: FastAPI, speaker: Speaker | None) -> None:
    """Adds the routes beside the chat: stopping a request, reading answers aloud.

    Args:
        app: The application.
        speaker: Reads answers aloud (MMS-TTS); None answers 404.
    """
    _speech_route(app, speaker)

    @app.post("/api/stop")
    def stop(request: Request) -> JSONResponse:
        """Asks the agent to stop: it does between steps, and the planner at once.

        Args:
            request: The request.

        Returns:
            ``{"stopping": true}``; the answer to the request then says it stopped.
        """
        request.state.visitor.stop.set()
        return JSONResponse({"stopping": True})


def _speech_route(app: FastAPI, speaker: Speaker | None) -> None:
    """Adds the route that reads answers aloud with the server's voice.

    Args:
        app: The application.
        speaker: Reads answers aloud (MMS-TTS); None answers 404.
    """

    @app.post("/api/speak")
    def speak(body: dict[str, str]) -> Response:
        """Reads an answer aloud with the server's voice.

        Args:
            body: ``{"text": ..., "language": ...}``.

        Returns:
            A WAV file; 404 if the server has no voice for that language.
        """
        try:
            audio = speaker(body.get("text", "")[:MAX_TEXT], body.get("language", "en")) \
                if speaker is not None else None
        except Exception as error:  # pylint: disable=broad-exception-caught
            # A voice that cannot load leaves the answer to the browser's voice.
            return JSONResponse({"error": f"The voice failed: {error}"}, status_code=503)
        if audio is None:
            return JSONResponse({"error": "No voice for this language."}, status_code=404)
        return Response(audio, media_type="audio/wav")


def _visitor_middleware(rooms: list[Room], by_id: dict[str, Room], visitors: Visitors,
                        meter: Meter | None,
                        uploads: VideoUploads | CloudVideoUploads | None) -> Callable[
                            [Request, Callable[[Request], Awaitable[Response]]],
                            Awaitable[Response]]:
    """Creates middleware for private room access and visitor cookies.

    Args:
        rooms: Public rooms, default first.
        by_id: Public and authorized private room registry.
        visitors: Per-browser scene state.
        meter: Optional public request limits.
        uploads: Optional persistent upload backend.

    Returns:
        The async HTTP middleware.
    """
    async def visitor_session(request: Request,
                              call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Attaches the visitor's session to the request, and sets their cookie.

        Args:
            request: The request.
            call_next: The route.

        Returns:
            The route's response.
        """
        rejected = await prepare_video_request(request, uploads, by_id)
        if rejected is not None:
            return rejected
        room = by_id.get(request.query_params.get("scene", ""), rooms[0])
        token, request.state.visitor = visitors.get(request.cookies.get(COOKIE), room.room_id)
        request.state.token, request.state.room = token, room
        response = await call_next(request)
        if request.cookies.get(COOKIE) != token:
            response.set_cookie(COOKIE, token, httponly=True, samesite="lax")
        remember_video_owner(request, response, uploads)
        _remember_owner(request, response, meter)
        return response

    return visitor_session


# pylint: disable-next=too-many-locals
def create_app(rooms: list[Room],
               agent_factory: Callable[[Session], Agent] | None = None,
               transcriber: Transcriber | None = None,
               meter: Meter | None = None,
               photographer: Photographer | None = None,
               guard: Guard | None = None, speaker: Speaker | None = None,
               uploads: VideoUploads | CloudVideoUploads | None = None) -> FastAPI:
    """Creates the web application for one or several rooms.

    The room of a request is given by its ``scene`` query parameter (the first
    room by default); each visitor works on their own copy of each room (see
    :mod:`pivot.server.visitors`).

    Args:
        rooms: The rooms, the default first; ids must be unique.
        agent_factory: Creates the agent of a session; None disables requests in words.
        transcriber: Turns spoken requests into text; None disables the microphone.
        meter: Prices requests and enforces a budget; None for a private server.
        photographer: Names the objects of a photo, sizes them and cuts one out;
            None disables objects from photos.
        guard: Screens requests in words before the agent sees them (Llama
            Prompt Guard, Llama Guard); None screens nothing.
        speaker: Reads answers aloud (MMS-TTS); None leaves it to the browser.
        uploads: Builds rooms from the visitors' own videos; None disables it.
            On a public server, only the owner's browser may send a video.

    Returns:
        The application.
    """
    app = FastAPI(title="PIVOT")
    # The page carries the point cloud: several megabytes, which compress well.
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    by_id = {room.room_id: room for room in rooms}
    visitors = Visitors({room.room_id: room.scene for room in rooms})
    pages = _render_pages(rooms, features := Features(
        interactive=True, chat=agent_factory is not None,
        voice=agent_factory is not None and transcriber is not None,
        photos=photographer is not None, uploads=uploads is not None,
        voices=tuple(MMS_VOICES) if speaker is not None else ()))

    app.middleware("http")(_visitor_middleware(rooms, by_id, visitors, meter, uploads))

    def key(request: Request) -> str:
        """Identifies the visitor of a request in the ledger.

        Args:
            request: The request.

        Returns:
            The visitor key; empty without a meter.
        """
        if meter is None:
            return ""
        if meter.is_owner(request.cookies.get(OWNER_COOKIE)):
            return OWNER
        return meter.visitor(request.headers, request.client.host if request.client else None)

    # Pages are compressed once: compressing several megabytes at each visit
    # took seconds on one CPU.
    compressed = {room_id: gzip.compress(page.encode(), compresslevel=6)
                  for room_id, page in pages.items()}

    add_video_routes(app, uploads, meter,
                     _room_adder(rooms, by_id, visitors, (pages, compressed), features),
                     partial(forget_video, rooms=by_id, pages=pages,
                             compressed=compressed, visitors=visitors))

    @app.get("/", response_class=HTMLResponse)
    def viewer(request: Request) -> Response:
        """Serves the viewer page of the requested room, compressed when the browser accepts it.

        Args:
            request: The request.

        Returns:
            The HTML page.
        """
        room_id = request.state.room.room_id
        if "gzip" not in request.headers.get("accept-encoding", ""):
            return HTMLResponse(pages[room_id],headers={"Cache-Control":"no-store"})
        return Response(compressed[room_id], media_type="text/html; charset=utf-8",
                        headers={"Content-Encoding": "gzip", "Vary": "Accept-Encoding",
                                 "Cache-Control":"no-store"})

    @app.get("/instructions", response_class=HTMLResponse)
    def moving_plan(request: Request) -> HTMLResponse:
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
        page = instructions_page(session.scene, instructions(initial, session.motions),
                                 request.state.room.title,
                                 save_pdf=request.query_params.get("pdf") == "1",
                                 removed=removed, added=added,
                                 access_routes=session.access_routes(),
                                 language=request.query_params.get("lang", "en"))
        # Always the visitor's latest moves, never a copy kept by the browser.
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

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

    @app.get("/api/access-routes")
    def access_routes(request: Request) -> JSONResponse:
        """Returns wheelchair routes verified in this visitor's current layout.

        Args:
            request: Request with its isolated visitor session.

        Returns:
            Current verified routes, without caching stale geometry.
        """
        return JSONResponse(request.state.visitor.session.access_routes(),
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/progress")
    def progress(request: Request) -> JSONResponse:
        """Says what the agent is doing on the visitor's request in words.

        Args:
            request: The request.

        Returns:
            ``{"steps": [{"tool": ..., "object_id": ...}, ...]}``, the tool
            calls of the current (or last) request so far.
        """
        return JSONResponse({"steps": list(request.state.visitor.progress)})

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

    @app.post("/api/photo")
    def photo(body: dict[str, Any], request: Request) -> JSONResponse:
        """Recognizes the object of a photo, for the visitor to place it.

        Args:
            body: ``{"image": "data:image/...;base64,..."}``.
            request: The request.

        Returns:
            What it is, its estimated size, its look and its picture; or why
            none was recognized.
        """
        if photographer is None:
            return JSONResponse({"error": "Objects from photos are not enabled."},
                                status_code=404)
        return _photo_reply(photographer, body, meter, key(request))

    @app.post("/api/spot")
    def spot(body: dict[str, Any], request: Request) -> JSONResponse:
        """Says whether a new object would fit at a spot, while the visitor picks it.

        Args:
            body: ``{"label", "size_m": [w, d, h], "x", "y", "yaw_deg"}``.
            request: The request.

        Returns:
            ``{"free": ..., "in_the_way": [ids]}``, and ``box`` where a wall
            object would hang; nothing is added.
        """
        size = tuple(float(v) for v in body["size_m"])
        trial, object_id, box = with_new_object(
            request.state.visitor.session.scene, str(body.get("label", "object")),
            (size[0], size[1], size[2]), float(body["x"]), float(body["y"]),
            float(body.get("yaw_deg", 0.0)))
        arrival = arrive(trial, object_id, box)
        # Hung on a wall, the ghost is drawn where it would hang.
        hung = {} if arrival.box is None else {"box": _ghost_box(arrival.box)}
        return JSONResponse({"free": arrival.placed, "in_the_way": arrival.in_the_way, **hung})

    _chat_side_routes(app, speaker)

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
    scene, points, colors = prepared_scene(scene_dir)
    points, colors = _subsample(points, colors, max_points)
    room_id = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return Room(room_id, title, scene, points, colors)


def _public_meter(config: MetaConfig | None) -> Meter:
    """Prices and limits the requests of a public server, from the environment.

    The budget comes from :meth:`Budget.from_env`; the ledger is kept at
    ``PIVOT_LEDGER`` (in memory if unset); the visitor key salt is
    ``PIVOT_VISITOR_SALT`` or derived from the API key; ``PIVOT_PROXY_HOPS``
    proxies are trusted; ``PIVOT_OWNER_KEY``, opened once as ``?owner=...``,
    lifts the daily limits of the owner's browser. Open-weight models cost nothing.

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
                 PHOTO_PRICE if api else 0.0,
                 os.environ.get("PIVOT_OWNER_KEY", ""))


def load_app(scene_dirs: list[str], env_file: Path | None = None, max_points: int | None = None,
             public: bool = False,
             uploads: VideoUploads | CloudVideoUploads | None = None) -> FastAPI:
    """Creates the application for rooms written by :mod:`pivot.build`.

    Args:
        scene_dirs: The rooms, as accepted by :func:`load_room`, the default first.
        env_file: API configuration file; requests in words are enabled if given.
        max_points: Most points of each cloud sent to the page; all if None.
        public: Whether the server is open to anyone: requests are then priced
            and limited (see :func:`_public_meter`).
        uploads: Optional remote backend; otherwise use the local environment setting.

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
                      photographer=_photographer(config) if api and config else None,
                      guard=guard_from_env() if config else None,
                      speaker=speaker_from_env(),
                      uploads=uploads if uploads is not None else _uploads_from_env())


def _uploads_from_env() -> VideoUploads | None:
    """Builds rooms from the visitors' videos if ``PIVOT_UPLOADS=1``.

    The pipeline (``scripts/run_video.sh``) needs a GPU job for the cameras
    and depth, and the Meta API; the videos and their rooms go to
    ``PIVOT_UPLOAD_DIR`` (a temporary folder by default).

    Returns:
        The uploads, or None when they are off.
    """
    if os.environ.get("PIVOT_UPLOADS") != "1":
        return None
    script = Path(__file__).resolve().parents[3] / "scripts" / "run_video.sh"
    folder = Path(os.environ.get("PIVOT_UPLOAD_DIR", Path(tempfile.gettempdir()) / "pivot_uploads"))
    return VideoUploads(folder, ["bash", str(script)])


def _photographer(config: MetaConfig) -> Photographer:
    """Builds what brings objects in from photos, with the Meta API.

    Args:
        config: API configuration.

    Returns:
        Muse Spark names and sizes the objects; SAM 3.1 cuts one out (its
        answers cached under ``PIVOT_CACHE``, or a temporary folder).
    """
    cache = Path(os.environ.get("PIVOT_CACHE", tempfile.gettempdir())) / "sam_photos"
    sam = SamClient(config, cache)
    return Photographer(partial(describe_objects, config),
                        partial(cut_object, segment=sam.segment_image))


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
