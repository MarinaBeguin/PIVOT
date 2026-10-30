"""HTTP server for the interactive viewer.

The page is the viewer of :mod:`pivot.viz.web` with its controls enabled. It
calls the same tools as the agent (:mod:`pivot.agent.tools`) on one shared
session, so a move made from the page is verified exactly like a move made by
the agent, and both can be undone. With ``--chat``, the page also takes
requests in words, answered by the Muse agent on that same session (the API
key is read from ``--env-file``), typed or spoken (:mod:`pivot.server.speech`).

Run locally::

    python -m pivot.server --scene-dir outputs/b20a261fdf --port 8000 [--chat]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from pivot.agent.loop import Agent
from pivot.agent.tools import TOOLS_BY_NAME, Session, call_tool
from pivot.geometry.types import FloatArray
from pivot.meta.client import MetaConfig
from pivot.planning.instructions import instructions
from pivot.planning.motion import MotionPlan
from pivot.scene.graph import Scene, carried_objects
from pivot.server.speech import Transcriber, transcriber_from_env
from pivot.viz.plan import instructions_page
from pivot.viz.pointcloud import ColorArray
from pivot.viz.web import render_viewer

_MUTATING = {"move_object", "make_room_and_move", "undo"}


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


# The routes are local functions: each counts as a local variable.
# pylint: disable-next=too-many-locals
def create_app(scene: Scene, points: FloatArray, colors: ColorArray,
               title: str = "PIVOT scene",
               agent_factory: Callable[[Session], Agent] | None = None,
               transcriber: Transcriber | None = None) -> FastAPI:
    """Creates the web application for one scene.

    Args:
        scene: The initial scene.
        points: (N, 3) point cloud shown in the viewer.
        colors: (N, 3) RGB colors.
        title: Page title.
        agent_factory: Creates the agent of a session; None disables requests in words.
        transcriber: Turns spoken requests into text; None disables the microphone.

    Returns:
        The application.
    """
    app = FastAPI(title="PIVOT")
    state: dict[str, Any] = {"session": Session(scene), "agent": None}
    page = render_viewer(scene, points, colors, title, interactive=True,
                         chat=agent_factory is not None,
                         voice=agent_factory is not None and transcriber is not None)

    @app.get("/", response_class=HTMLResponse)
    def viewer() -> str:
        """Serves the viewer page.

        Returns:
            The HTML page.
        """
        return page

    @app.get("/instructions", response_class=HTMLResponse)
    def moving_plan() -> str:
        """Serves the moving instructions of the moves made so far.

        Returns:
            A printable page: plan from above and numbered steps.
        """
        session: Session = state["session"]
        initial = session.history[0] if session.history else session.scene
        return instructions_page(session.scene, instructions(initial, session.motions), title)

    @app.get("/api/objects")
    def objects() -> list[dict[str, Any]]:
        """Lists the current objects.

        Returns:
            The objects, sorted by id.
        """
        return _objects(state["session"])

    @app.post("/api/tools/{name}")
    def tool(name: str, arguments: dict[str, Any]) -> JSONResponse:
        """Calls a geometric tool on the shared session.

        Args:
            name: Tool name, see :data:`pivot.agent.tools.TOOLS`.
            arguments: Tool arguments.

        Returns:
            The tool result; moves and undos also carry the updated objects.
        """
        if name not in TOOLS_BY_NAME:
            return JSONResponse({"error": f"Unknown tool {name!r}."}, status_code=404)
        session = state["session"]
        result: dict[str, Any] = json.loads(call_tool(session, name, json.dumps(arguments)))
        if name in _MUTATING:
            result["objects"] = _objects(session)
        return JSONResponse(result)

    @app.post("/api/chat")
    def chat(request: dict[str, str]) -> JSONResponse:
        """Answers a request in words with the agent, on the shared session.

        Args:
            request: ``{"message": ...}``.

        Returns:
            The answer, the tools called, the motions executed (to animate)
            and the updated objects.
        """
        if agent_factory is None:
            return JSONResponse({"error": "Requests in words are not enabled."},
                                status_code=404)
        session: Session = state["session"]
        agent: Agent = state["agent"] or agent_factory(session)
        state["agent"] = agent
        before, count = session.scene, len(session.motions)
        try:
            run = agent.ask(request.get("message", ""))
        except Exception as error:  # pylint: disable=broad-exception-caught
            # An API failure must reach the page, not break the server.
            return JSONResponse({"error": f"The agent could not answer: {error}"})
        return JSONResponse({
            "answer": run.answer, "finished": run.finished,
            "steps": [{"tool": s.tool, "arguments": s.arguments} for s in run.steps],
            "motions": _motions(before, session.motions[count:]),
            "objects": _objects(session)})

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
        labels = sorted({o.label for o in state["session"].scene.objects.values()})
        try:
            text = transcriber(await request.body(), labels)
        except Exception as error:  # pylint: disable=broad-exception-caught
            # A transcription failure must reach the page, not break the server.
            return JSONResponse({"error": f"Could not transcribe: {error}"})
        return JSONResponse({"text": text})

    @app.post("/api/reset")
    def reset() -> list[dict[str, Any]]:
        """Restores the initial scene and starts a new conversation.

        Returns:
            The objects, sorted by id.
        """
        state["session"] = Session(scene)
        state["agent"] = None
        return _objects(state["session"])

    return app


def load_app(scene_dir: Path, env_file: Path | None = None) -> FastAPI:
    """Creates the application for a directory written by :mod:`pivot.build`.

    Args:
        scene_dir: Directory holding ``scene.json`` and ``cloud.npz``.
        env_file: API configuration file; requests in words are enabled if given.

    Returns:
        The application.
    """
    cloud = np.load(scene_dir / "cloud.npz")
    config = MetaConfig.from_env(env_file) if env_file else None
    # A conversation may rearrange a whole room: many more steps than one move.
    factory = partial(Agent, config=config, max_steps=40) if config else None
    return create_app(Scene.load(scene_dir / "scene.json"), cloud["points"], cloud["colors"],
                      title=f"PIVOT - {scene_dir.name}", agent_factory=factory,
                      transcriber=transcriber_from_env(config) if config else None)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point.

    Args:
        argv: Command-line arguments; ``sys.argv`` if None.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--chat", action="store_true", help="answer requests in words")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args(argv)

    # Imported here: only needed to serve, not to build or test the app.
    import uvicorn  # pylint: disable=import-outside-toplevel

    uvicorn.run(load_app(args.scene_dir, args.env_file if args.chat else None),
                host=args.host, port=args.port)
