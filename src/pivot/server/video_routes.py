"""Video HTTP routes and durable browser ownership, shared by local and Modal backends."""
from __future__ import annotations

import re
import secrets
import tempfile
from collections.abc import Callable, Collection, MutableMapping
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from pivot.server.cloud_uploads import CloudVideoUploads, UploadError
from pivot.server.limits import Meter
from pivot.server.uploads import VideoUploads
from pivot.server.visitors import Visitors

OWNER_COOKIE = "pivot_owner"


def add_video_routes(app: FastAPI, uploads: VideoUploads | CloudVideoUploads | None,
                     meter: Meter | None,
                  on_ready: Callable[[str, Path], None],
                  on_deleted: Callable[[str], None]) -> None:
    """Adds the routes that build rooms from the visitors' videos.

    Args:
        app: The application.
        uploads: Runs the pipeline on the videos; None adds no route.
        meter: On a public server, only the owner's browser may send a video.
        on_ready: Serves a room once built, from the job's id and its scene.
        on_deleted: Removes a deleted room from the web caches.
    """
    if uploads is None:
        return
    uploads.on_ready = on_ready

    @app.post("/api/video")
    async def video(request: Request) -> JSONResponse:
        """Starts building a room from a video sent as the request's body.

        Args:
            request: Its ``X-Filename`` header names the file.

        Returns:
            ``{"job": id}``, or why the video is refused.
        """
        if meter is not None and not uploads.allow_public and not meter.is_owner(
                request.cookies.get(OWNER_COOKIE)):
            return JSONResponse({"error": "Video uploads are not open on this server."},
                                status_code=403)
        owner = request.state.upload_owner
        budget_key = meter.visitor(request.headers, request.client.host if request.client
                                   else None) if meter else owner
        if isinstance(uploads, CloudVideoUploads):
            try:
                await run_in_threadpool(uploads.check_admission, owner, budget_key)
            except UploadError as error:
                return JSONResponse({"error": str(error)}, status_code=error.status_code)
        with tempfile.TemporaryDirectory(prefix="pivot-upload-") as directory:
            path = Path(directory) / "input"
            size = 0
            with path.open("wb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > uploads.max_bytes:
                        return JSONResponse({"error": "Video exceeds this server's size limit."},
                                            status_code=413)
                    output.write(chunk)
            if not size:
                return JSONResponse({"error": "Please choose a video."}, status_code=422)
            try:
                job = await run_in_threadpool(uploads.start_file, path,
                    request.headers.get("x-filename", "video.mp4"), owner, budget_key)
            except UploadError as error:
                return JSONResponse({"error": str(error)}, status_code=error.status_code)
        return JSONResponse({"job": job})

    @app.get("/api/video/config")
    def video_config() -> JSONResponse:
        """Returns upload bounds and cloud processing scope.

        Returns:
            Public upload configuration.
        """
        remote = isinstance(uploads, CloudVideoUploads)
        info: dict[str, Any] = {"max_bytes": uploads.max_bytes, "cloud": remote}
        if isinstance(uploads, CloudVideoUploads):
            info.update(max_seconds=uploads.limits.max_seconds,
                        retention_hours=uploads.limits.retention / 3600)
        return JSONResponse(info)

    @app.get("/api/video/{job_id}")
    def video_status(job_id: str, request: Request) -> JSONResponse:
        """Says how the room of a video is coming along.

        Args:
            job_id: The job.
            request: Request containing the browser capability.

        Returns:
            Its state, the last lines of its log, and its room once built.
        """
        status = uploads.status(job_id, request.state.upload_owner)
        if status is None:
            return JSONResponse({"error": "No such video."}, status_code=404)
        return JSONResponse(status)

    @app.delete("/api/video/{job_id}")
    def delete_video(job_id: str, request: Request) -> JSONResponse:
        """Deletes only a job belonging to this browser.

        Args:
            job_id: Uploaded video job.
            request: Browser request carrying the upload cookie.

        Returns:
            Confirmation or an indistinguishable not-found response.
        """
        if not uploads.delete(job_id, request.state.upload_owner):
            return JSONResponse({"error": "No such video."}, status_code=404)
        on_deleted(job_id)
        return JSONResponse({"deleted": True})


async def prepare_video_request(request: Request, uploads: VideoUploads | CloudVideoUploads | None,
                                known_rooms: Collection[str]) -> JSONResponse | None:
    """Checks private-room ownership before any page, object or tool route.

    Args:
        request: Incoming browser request.
        uploads: Local or remote upload backend.
        known_rooms: Room ids currently loaded by the web process.

    Returns:
        A not-found response for unauthorized/unready private rooms, else None.
    """
    owner = request.cookies.get("pivot_video_owner", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,64}", owner):
        owner = secrets.token_urlsafe(32)
    request.state.upload_owner = owner
    requested = request.query_params.get("scene", "")
    if re.fullmatch(r"video-[0-9a-f]{32}", requested):
        if uploads is None or not await run_in_threadpool(uploads.owns, requested, owner):
            return JSONResponse({"error": "No such room."}, status_code=404)
        if requested not in known_rooms:
            status = await run_in_threadpool(uploads.status, requested, owner)
            if status is None or status.get("state") != "ready":
                return JSONResponse({"error": "The room is not ready."}, status_code=404)
    return None


def remember_video_owner(request: Request, response: Response,
                         uploads: VideoUploads | CloudVideoUploads | None) -> None:
    """Preserves the upload capability across web process restarts.

    Args:
        request: Incoming browser request.
        response: Outgoing response.
        uploads: Active backend, or None.
    """
    owner = request.state.upload_owner
    if uploads is not None and request.cookies.get("pivot_video_owner") != owner:
        response.set_cookie("pivot_video_owner", owner, httponly=True,
                            samesite="lax", secure=request.url.scheme == "https",
                            max_age=7 * 86400)


def forget_video(room_id: str, rooms: MutableMapping[str, Any], pages: MutableMapping[str, Any],
                 compressed: MutableMapping[str, Any], visitors: Visitors) -> None:
    """Removes a deleted room and every cached visitor copy.

    Args:
        room_id: Private room to remove.
        rooms: Web room registry.
        pages: Plain rendered pages.
        compressed: Compressed rendered pages.
        visitors: Visitor scene/undo state registry.
    """
    rooms.pop(room_id, None)
    pages.pop(room_id, None)
    compressed.pop(room_id, None)
    visitors.remove_room(room_id)
