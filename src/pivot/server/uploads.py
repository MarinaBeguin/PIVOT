"""Rooms from the visitors' own videos, built while they wait.

A visitor sends a video filmed with any phone; it is saved, and the whole
pipeline runs on it in the background (``scripts/run_video.sh``: cameras and
depth by MapAnything, private parts blurred, objects listed and tracked, the
scene built). The page polls the job; once the scene is built, the server
loads it as a new room, which the visitor opens like the others.

Building a room takes a GPU and tens of minutes: the public demo keeps this
off unless enabled (``PIVOT_UPLOADS=1``), and then only for its owner's key
(see :class:`pivot.server.limits.Meter`), which can be given to judges.
"""

from __future__ import annotations

import os
import re
import secrets
import shutil
import tempfile
import subprocess
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

# Largest video accepted, in bytes: about two minutes of 4K from a phone.
MAX_VIDEO_BYTES = 500_000_000
# Lines of the job's log shown to the visitor.
_LOG_LINES = 6


@dataclass
class VideoJob:
    """One video being turned into a room.

    Attributes:
        job_id: Its name, also the new room's id.
        folder: Where the video, the log and the outputs go.
        process: The pipeline, while it runs.
        state: ``running``, ``ready`` or ``failed``.
    """

    job_id: str
    folder: Path
    process: subprocess.Popen[bytes] | None = None
    state: str = "running"
    owner: str = ""

    @property
    def scene_dir(self) -> Path:
        """Where the pipeline writes the scene."""
        return self.folder / "out" / "room" / "scene"

    @property
    def log(self) -> Path:
        """The pipeline's output."""
        return self.folder / "pipeline.log"


@dataclass
class VideoUploads:
    """Runs the pipeline on uploaded videos and hands over the rooms they become.

    Attributes:
        folder: Where the jobs' folders go.
        runner: The command run on a video and an output folder.
        on_ready: Called with a job's id and its scene directory, once built.
        jobs: The jobs, by id.
    """

    allow_public: ClassVar[bool] = False
    max_bytes: ClassVar[int] = MAX_VIDEO_BYTES
    folder: Path
    runner: list[str]
    on_ready: Callable[[str, Path], None] = lambda job_id, scene: None
    jobs: dict[str, VideoJob] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self, video: bytes, filename: str, owner: str = "") -> str:
        """Starts a local job from bytes.

        Args:
            video: Encoded media.
            filename: Filename used for the extension.
            owner: Browser upload capability.

        Returns:
            Job id.
        """
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input"
            source.write_bytes(video)
            return self.start_file(source, filename, owner)

    def start_file(self, source: Path, filename: str, owner: str,
                   budget_key: str = "") -> str:
        """Starts a local job from a spooled file.

        Args:
            source: File to copy.
            filename: Its original extension.
            owner: Browser upload capability.
            budget_key: Reserved for remote admission limits.

        Returns:
            Job id.
        """
        del budget_key
        job_id = "video-" + secrets.token_hex(16)
        folder = self.folder / job_id
        folder.mkdir(parents=True, exist_ok=True)
        suffix = re.sub(r"[^a-z0-9.]", "", Path(filename).suffix.lower()) or ".mp4"
        path = folder / f"video{suffix}"
        shutil.copyfile(source, path)
        job = VideoJob(job_id, folder, owner=owner)
        with job.log.open("wb") as log:
            # The pipeline builds the scene; this server serves it (no second server).
            job.process = subprocess.Popen(  # pylint: disable=consider-using-with
                [*self.runner, str(path), str(folder / "out")], stdout=log,
                stderr=subprocess.STDOUT, env={**os.environ, "PIVOT_NO_SERVE": "1"})
        with self._lock:
            self.jobs[job_id] = job
        return job_id

    def status(self, job_id: str, owner: str = "") -> dict[str, Any] | None:
        """Says how a job is going, and hands over its room once built.

        Args:
            job_id: The job.
            owner: Browser upload capability.

        Returns:
            ``{"state", "log"}``, with ``room`` once ready; None for an unknown job.
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None or not secrets.compare_digest(job.owner, owner):
                return None
            if job.state == "running" and job.process is not None \
                    and job.process.poll() is not None:
                built = job.process.returncode == 0 and (job.scene_dir / "scene.json").exists()
                job.state = "ready" if built else "failed"
                if built:
                    self.on_ready(job_id, job.scene_dir)
        lines = job.log.read_text(encoding="utf-8", errors="replace").splitlines() \
            if job.log.exists() else []
        answer: dict[str, Any] = {"state": job.state, "log": lines[-_LOG_LINES:]}
        if job.state == "ready":
            answer["room"] = job_id
        return answer

    def owns(self, job_id: str, owner: str) -> bool:
        """Checks whether a local job belongs to a browser.

        Args:
            job_id: Job id.
            owner: Browser capability.

        Returns:
            Whether access is allowed.
        """
        with self._lock:
            job = self.jobs.get(job_id)
            return job is not None and secrets.compare_digest(job.owner, owner)

    def delete(self, job_id: str, owner: str) -> bool:
        """Revokes and removes an owned local job.

        Args:
            job_id: Job id.
            owner: Browser capability.

        Returns:
            Whether a job was removed.
        """
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None or not secrets.compare_digest(job.owner, owner):
                return False
            self.jobs.pop(job_id)
            if job.process is not None and job.process.poll() is None:
                job.process.terminate()
        shutil.rmtree(job.folder, ignore_errors=True)
        return True
