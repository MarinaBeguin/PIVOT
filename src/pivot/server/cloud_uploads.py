"""Persistent, visitor-scoped admission and status for remote video jobs.

The web container is the single admission writer. Workers write only their own
status files. Upload quotas are separate from the conversational API ledger.
"""
from __future__ import annotations

import hashlib
import json
import math
import secrets
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar


class UploadError(ValueError):
    """An upload was rejected before processing, or its service is unavailable."""

    def __init__(self, message: str, status_code: int = 422):
        super().__init__(message)
        self.status_code = status_code


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Replaces a metadata file atomically.

    Args:
        path: Destination file.
        data: Serializable metadata.
    """
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data), encoding='utf-8')
    temporary.replace(path)


def write_state(folder: Path, state: str, message: str) -> None:
    """Writes a job state atomically without reviving a deleted job.

    Args:
        folder: Job directory.
        state: Machine-readable state.
        message: Public, controlled progress message (never raw subprocess logs).
    """
    if (folder / 'deleted').exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / ('status-' + secrets.token_hex(6) + '.tmp')
    temporary.write_text(json.dumps({'state': state, 'message': message}), encoding='utf-8')
    temporary.replace(folder / 'status.json')


def validate_video(path: Path, max_seconds: float) -> None:
    """Checks video duration and dimensions before spending GPU time.

    Args:
        path: Uploaded media file.
        max_seconds: Largest accepted duration.

    Raises:
        UploadError: If the media is invalid or exceeds the configured limits.
    """
    try:
        result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                                 'format=duration:stream=codec_type,width,height',
                                 '-of', 'json', str(path)], capture_output=True, text=True,
                                timeout=15, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        videos = [v for v in data['streams'] if v.get('codec_type') == 'video']
        valid = videos and all(32 <= int(v.get('width', 0)) <= 4096 and
                               32 <= int(v.get('height', 0)) <= 4096 for v in videos)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as error:
        raise UploadError('The file is not a readable video with a known duration.') from error
    if not valid or not math.isfinite(duration) or not 0 < duration <= max_seconds:
        raise UploadError(f'Use a video up to {max_seconds:g} seconds '
                          'and 4096 pixels per side.')



@dataclass(frozen=True)
class UploadLimits:
    """Admission and retention bounds, independent of chat budgets.

    Attributes:
        max_bytes: Maximum incoming bytes.
        max_seconds: Maximum media duration.
        daily: Global submissions per UTC day, including deleted/failed jobs.
        per_visitor: Submissions per IP-derived visitor key and UTC day.
        active: Most outstanding jobs, including queued jobs.
        retention: Seconds before owner access expires and files are cleaned.
        timeout: Longest combined processing time before a job is failed.
    """
    max_bytes: int = 100_000_000
    max_seconds: float = 120
    daily: int = 4
    per_visitor: int = 1
    active: int = 1
    retention: int = 86400
    timeout: int = 1800

    def __post_init__(self) -> None:
        """Rejects configurations that cannot bound processing."""
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError('Upload duration must be finite and positive.')
        if min(self.max_bytes, self.daily, self.per_visitor, self.active,
               self.retention, self.timeout) <= 0:
            raise ValueError('Upload bounds must be positive.')


@dataclass
class CloudVideoUploads:
    """Admission controller for a persistent store and remote workers.

    One web container must own admission; its thread lock serializes requests.
    Worker callbacks and volume operations are injected so CPU tests need no
    Modal account, API credential or GPU.

    Attributes:
        folder: Persistent jobs and quota ledger.
        submit: Starts a job by id, returning a durable remote call id.
        poll: Returns running, finished or failed for a remote call chain.
        cancel: Cancels a remote call.
        refresh: Refreshes the mounted store after other workers' commits.
        commit: Publishes local store changes.
        limits: Independent upload bounds.
        on_ready: Registers an authorized completed room in this web process.
    """
    folder: Path
    submit: Callable[[str], str]
    poll: Callable[[str], str]
    cancel: Callable[[str], None]
    refresh: Callable[[], None] = lambda: None
    commit: Callable[[], None] = lambda: None
    limits: UploadLimits = field(default_factory=UploadLimits)
    on_ready: Callable[[str, Path], None] = lambda job_id, scene: None
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _published: set[str] = field(default_factory=set, init=False)
    allow_public: ClassVar[bool] = True

    @property
    def max_bytes(self) -> int:
        """Returns the upload byte bound."""
        return self.limits.max_bytes

    def _meta(self, job_id: str) -> dict[str, Any] | None:
        """Reads metadata for a syntactically valid id.

        Args:
            job_id: Public job identifier.

        Returns:
            Metadata, or None for an unknown, expired or deleted job.
        """
        if not job_id.startswith('video-') or len(job_id) != 38 \
                or any(c not in '0123456789abcdef' for c in job_id[6:]):
            return None
        folder = self.folder / job_id
        if (folder / 'deleted').exists() or not (folder / 'meta.json').is_file():
            return None
        meta: dict[str, Any] = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
        return meta if float(meta['expires']) > time.time() else None

    def owns(self, job_id: str, owner: str) -> bool:
        """Checks a durable browser capability without relying on in-memory visitors.

        Args:
            job_id: Job to open.
            owner: Browser's random upload cookie.

        Returns:
            Whether the unexpired job belongs to that browser.
        """
        with self._lock:
            self.refresh()
            meta = self._meta(job_id)
            return bool(meta and secrets.compare_digest(str(meta['owner']),
                         hashlib.sha256(owner.encode()).hexdigest()))

    def _status(self, job_id: str, meta: dict[str, Any]) -> dict[str, Any]:
        """Reads worker status and reconciles crashed remote calls.

        Args:
            job_id: Job id.
            meta: Immutable admission metadata and initial call id.

        Returns:
            Worker state and safe progress message.
        """
        folder = self.folder / job_id
        path = folder / 'status.json'
        state: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {
            'state': 'queued', 'message': 'Waiting for processing.'}
        if state['state'] in ('queued', 'running'):
            call_id = str(meta.get('call_id', ''))
            finished = self.poll(call_id) if call_id else 'running'
            if finished != 'running':
                self.refresh()
                state = json.loads(path.read_text()) if path.exists() else state
            if state['state'] in ('queued', 'running') and (finished != 'running' or
                    time.time() - float(meta['created']) > self.limits.timeout):
                if call_id:
                    self.cancel(call_id)
                write_state(folder, 'failed', 'Processing stopped or exceeded its time limit.')
                self.commit()
                state = json.loads(path.read_text())
        return state

    def _capacity(self, owner: str, budget_key: str) -> tuple[Path, str, dict[str, Any], str]:
        """Checks admission without consuming quota.

        Args:
            owner: Browser capability.
            budget_key: Stable public visitor/IP key.

        Returns:
            Ledger, UTC day, counts and hashed visitor key.

        Raises:
            UploadError: If quotas or outstanding jobs prevent admission.
        """
        ledger = self.folder / 'usage.json'
        try:
            usage = json.loads(ledger.read_text()) if ledger.exists() else {}
        except (OSError, ValueError) as error:
            raise UploadError('Video quota storage is unavailable.', 503) from error
        day = time.strftime('%Y-%m-%d', time.gmtime())
        counts = usage.get(day, {'total': 0, 'visitors': {}})
        visitor = hashlib.sha256((budget_key or owner).encode()).hexdigest()
        if counts['total'] >= self.limits.daily or \
                counts['visitors'].get(visitor, 0) >= self.limits.per_visitor:
            raise UploadError('Video processing quota reached for today.', 429)
        active = 0
        for entry in self.folder.glob('video-*'):
            meta = self._meta(entry.name)
            if meta and self._status(entry.name, meta)['state'] in ('queued', 'running'):
                active += 1
        if active >= self.limits.active:
            raise UploadError('A reconstruction is already in progress. Please try later.', 429)
        return ledger, day, counts, visitor

    def check_admission(self, owner: str, budget_key: str) -> None:
        """Rejects exhausted quotas before reading a potentially large HTTP body.

        Args:
            owner: Browser capability.
            budget_key: Stable public visitor/IP key.
        """
        with self._lock:
            self.refresh()
            self._capacity(owner, budget_key)

    def _reserve(self, owner: str, budget_key: str) -> None:
        """Reserves capacity; the caller holds the lock and commits before spawning.

        Args:
            owner: Browser capability.
            budget_key: Stable public visitor/IP key.
        """
        ledger, day, counts, visitor = self._capacity(owner, budget_key)
        counts['total'] += 1
        counts['visitors'][visitor] = counts['visitors'].get(visitor, 0) + 1
        write_json(ledger, {day: counts})

    def start_file(self, source: Path, filename: str, owner: str,
                   budget_key: str = '') -> str:
        """Validates and admits media, recording quota before remote submission.

        Args:
            source: Spool file (not loaded into RAM).
            filename: Name used only to choose a safe extension.
            owner: Random browser capability.
            budget_key: Stable public visitor/IP key; never stored in plaintext.

        Returns:
            New job id.

        Raises:
            UploadError: If media, quotas or the remote service reject the job.
        """
        if source.stat().st_size > self.max_bytes:
            raise UploadError('The video exceeds the upload size limit.', 413)
        validate_video(source, self.limits.max_seconds)
        with self._lock:
            self.refresh()
            self.folder.mkdir(parents=True, exist_ok=True)
            self._reserve(owner, budget_key)
            job_id = 'video-' + secrets.token_hex(16)
            job = self.folder / job_id
            job.mkdir()
            suffix = Path(filename).suffix.lower()
            if suffix not in ('.mp4', '.mov', '.m4v', '.webm', '.mkv', '.avi'):
                suffix = '.mp4'
            shutil.copyfile(source, job / ('input' + suffix))
            meta = {'owner': hashlib.sha256(owner.encode()).hexdigest(),
                    'created': time.time(), 'expires': time.time() + self.limits.retention,
                    'input': 'input' + suffix}
            path = job / 'meta.json'
            write_json(path, meta)
            write_state(job, 'queued', 'Waiting for a GPU. You can return to this page later.')
            self.commit()
            try:
                meta['call_id'] = self.submit(job_id)
                write_json(path, meta)
                self.commit()
            except Exception as error:  # pylint: disable=broad-exception-caught
                write_state(job, 'failed', 'Could not start remote processing.')
                self.commit()
                raise UploadError('Video processing could not be started.', 503) from error
            return job_id

    def status(self, job_id: str, owner: str = '') -> dict[str, Any] | None:
        """Returns safe state and registers the room for its owner only.

        Args:
            job_id: Job id.
            owner: Browser capability.

        Returns:
            State, or None if this browser cannot access the job.
        """
        with self._lock:
            if not self.owns(job_id, owner):
                return None
            meta = self._meta(job_id)
            assert meta is not None
            state = self._status(job_id, meta)
            answer: dict[str, Any] = {'state': 'running' if state['state'] == 'queued'
                                     else state['state'], 'log': [state['message']]}
            scene = self.folder / job_id / 'out' / 'room' / 'scene'
            if state['state'] == 'ready':
                if not all((scene / name).is_file() for name in ('scene.json', 'cloud.npz')):
                    return {'state': 'failed', 'log': ['The scene output is incomplete.']}
                if job_id not in self._published:
                    self.on_ready(job_id, scene)
                    self._published.add(job_id)
                answer['room'] = job_id
            return answer

    def delete(self, job_id: str, owner: str) -> bool:
        """Revokes access, cancels active calls and removes job payloads.

        Args:
            job_id: Job to remove.
            owner: Browser capability.

        Returns:
            Whether an owned job was found and revoked.
        """
        with self._lock:
            if not self.owns(job_id, owner):
                return False
            meta = self._meta(job_id)
            assert meta is not None
            folder = self.folder / job_id
            (folder / 'deleted').touch()
            self.commit()
            calls = [str(meta.get('call_id', ''))]
            next_call = folder / 'continuation.json'
            if next_call.exists():
                calls.append(str(json.loads(next_call.read_text())['call_id']))
            for call_id in filter(None, calls):
                self.cancel(call_id)
            for item in folder.iterdir():
                if item.name not in ('deleted', 'meta.json'):
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            self.commit()
            self._published.discard(job_id)
            return True
