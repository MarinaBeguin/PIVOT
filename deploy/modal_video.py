"""Modal video workers: GPU capture followed by CPU/API scene processing.

Imported by modal_app; no worker is deployed until the user deploys that app.
"""
from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import modal

from pivot.server.cloud_uploads import CloudVideoUploads, UploadLimits, write_state

ROOT = Path(__file__).resolve().parents[1]
STORE = Path('/videos')


def _run(command: list[str], *, timeout: int, env: dict[str, str] | None = None) -> None:
    """Runs a stage and kills its whole subprocess group on timeout."""
    with subprocess.Popen(command, env=env, start_new_session=True) as process:
        try:
            result = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
        if result:
            raise subprocess.CalledProcessError(result, command)


def attach_uploads(app: modal.App, cpu_image: modal.Image,
                   secret: modal.Secret) -> tuple[Any, modal.Volume]:
    """Registers workers and returns a web-side backend factory and its volume."""
    namespace = os.environ.get('PIVOT_UPLOAD_NAMESPACE', 'pivot')
    if re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9-]{0,30}', namespace) is None:
        raise ValueError('Invalid upload storage namespace')
    volume = modal.Volume.from_name(namespace + '-video-jobs', create_if_missing=True)
    models = modal.Volume.from_name(namespace + '-video-models', create_if_missing=True)
    frames = int(os.environ.get('PIVOT_UPLOAD_FRAMES', '150'))
    max_kinds = int(os.environ.get('PIVOT_UPLOAD_OBJECT_KINDS', '20'))
    limits = UploadLimits(
        max_bytes=int(os.environ.get('PIVOT_UPLOAD_MAX_BYTES', '100000000')),
        max_seconds=float(os.environ.get('PIVOT_UPLOAD_MAX_SECONDS', '120')),
        daily=int(os.environ.get('PIVOT_UPLOAD_DAILY', '4')),
        per_visitor=int(os.environ.get('PIVOT_UPLOAD_PER_VISITOR', '1')),
        active=int(os.environ.get('PIVOT_UPLOAD_ACTIVE', '1')),
        retention=int(os.environ.get('PIVOT_UPLOAD_RETENTION', '86400')))
    if not 1 <= frames <= 150 or not 1 <= max_kinds <= 20:
        raise ValueError('Upload frames must be 1..150 and object kinds 1..20.')
    gpu_image = (
        modal.Image.debian_slim(python_version='3.12')
        .apt_install('ffmpeg', 'git', 'libglib2.0-0', 'libgl1')
        .add_local_file(ROOT / 'docker/map-constraints.txt', '/opt/map-constraints.txt', copy=True)
        .env({'PIP_CONSTRAINT': '/opt/map-constraints.txt',
              'HF_HOME': '/cache/huggingface', 'TORCH_HOME': '/cache/torch',
              'PYTHONUNBUFFERED': '1'})
        .run_commands('pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 '
                      '--index-url https://download.pytorch.org/whl/cu126')
        .add_local_file(ROOT / 'pyproject.toml', '/root/pivot/pyproject.toml', copy=True)
        .add_local_file(ROOT / 'README.md', '/root/pivot/README.md', copy=True)
        .add_local_dir(ROOT / 'src', '/root/pivot/src', copy=True,
                       ignore=['**/__pycache__/**'])
        .run_commands("pip install '/root/pivot' 'scipy<1.17' "
                      "'git+https://github.com/facebookresearch/map-anything.git"
                      "@3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9'", 'pip check')
        .add_local_dir(ROOT / 'deploy', '/root/pivot/deploy', copy=True, ignore=['**/__pycache__/**'])
        .env({'PYTHONPATH': '/root/pivot/src:/root/pivot/deploy'}))

    def folder_for(job_id: str) -> Path:
        if re.fullmatch(r'video-[0-9a-f]{32}', job_id) is None:
            raise ValueError('Invalid job id')
        folder = STORE / job_id
        if not (folder / 'meta.json').is_file() or (folder / 'deleted').exists():
            raise ValueError('Job is absent or deleted')
        return folder

    def cancelled(folder: Path) -> bool:
        # Publish closed stage files before refreshing the other container's tombstone.
        volume.commit()
        volume.reload()
        if (folder / 'deleted').exists():
            for item in folder.iterdir():
                if item.name not in ('deleted', 'meta.json'):
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
            volume.commit()
            return True
        return False

    @app.function(name='video_scene', serialized=True, image=cpu_image, secrets=[secret],
                  volumes={'/videos': volume}, cpu=4, memory=16384,
                  timeout=900, max_containers=1, retries=0)
    def scene_job(job_id: str) -> dict[str, str]:
        volume.reload()
        folder = folder_for(job_id)
        try:
            write_state(folder, 'running', 'Blurring detected private regions with the Meta API.')
            volume.commit()
            env = {**os.environ, 'PIVOT_BACKEND': 'api', 'PIVOT_NO_SERVE': '1',
                   'PIVOT_PYTHON': sys.executable, 'PIVOT_SCRATCH': '/tmp/pivot',
                   'PIVOT_CACHE': str(folder / 'cache'), 'NUMBA_CACHE_DIR': '/tmp/pivot-numba',
                   'PIVOT_MAX_OBJECTS_API': str(max_kinds), 'PIVOT_REQUIRE_INVENTORY': '1', 'PIVOT_SAM_WORKERS': '4'}
            capture = folder / 'out' / 'capture'
            _run([sys.executable, '-m', 'pivot.privacy', '--capture', str(capture)], env=env, timeout=400)
            if cancelled(folder):
                return {'state': 'deleted'}
            write_state(folder, 'running', 'Identifying objects and building the 3D scene.')
            volume.commit()
            _run(['bash', '/root/pivot/scripts/run_api.sh', str(capture),
                            str(capture / 'video.mp4'), str(folder / 'out' / 'room')],
                           env={**env, 'PIVOT_VIDEO_STRIDE': '1'}, timeout=450)
            scene = folder / 'out' / 'room' / 'scene'
            if not all((scene / name).is_file() for name in ('scene.json', 'cloud.npz')):
                raise RuntimeError('Missing scene output')
            if cancelled(folder):
                return {'state': 'deleted'}
            write_state(folder, 'ready', 'Your room is ready.')
            volume.commit()
            return {'state': 'ready'}
        except Exception:
            write_state(folder, 'failed', 'Scene generation failed. Please try another video.')
            volume.commit()
            raise

    @app.function(name='video_capture', serialized=True, image=gpu_image, gpu='L40S',
                  volumes={'/videos': volume, '/cache': models},
                  cpu=4, memory=65536, timeout=900, max_containers=1, retries=0)
    def capture_job(job_id: str) -> dict[str, str]:
        volume.reload()
        folder = folder_for(job_id)
        meta = json.loads((folder / 'meta.json').read_text())
        try:
            write_state(folder, 'running', 'Estimating cameras and depth on a Modal GPU.')
            volume.commit()
            _run([sys.executable, '-m', 'pivot.reconstruction.video_capture',
                            '--video', str(folder / meta['input']),
                            '--out', str(folder / 'out' / 'capture'), '--frames', str(frames)],
                           timeout=800)
            models.commit()
            if cancelled(folder):
                return {'state': 'deleted'}
            write_state(folder, 'running', 'Reconstruction done; waiting for scene processing.')
            volume.commit()
            call = scene_job.spawn(job_id)
            (folder / 'continuation.json').write_text(json.dumps({'call_id': call.object_id}))
            volume.commit()
            return {'next_call_id': call.object_id}
        except Exception:
            write_state(folder, 'failed', 'Video reconstruction failed. Please try another video.')
            volume.commit()
            raise

    def poll(call_id: str) -> str:
        try:
            for _ in range(3):
                result = modal.FunctionCall.from_id(call_id).get(timeout=0)
                if not isinstance(result, dict) or 'next_call_id' not in result:
                    return 'finished'
                call_id = str(result['next_call_id'])
            return 'failed'
        except TimeoutError:
            return 'running'
        except Exception:
            return 'failed'

    def cancel(call_id: str) -> None:
        try:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        except Exception:
            # Terminal calls cannot be cancelled; access is already revoked by the tombstone.
            pass

    @app.function(name='video_cleanup', serialized=True, image=cpu_image, volumes={'/videos': volume},
                  schedule=modal.Cron('0 * * * *'), timeout=300, max_containers=1)
    def cleanup() -> None:
        volume.reload()
        for folder in STORE.glob('video-*'):
            path = folder / 'meta.json'
            if path.is_file():
                meta = json.loads(path.read_text())
                if (folder / 'deleted').exists() or float(meta['expires']) <= time.time():
                    continuation = folder / 'continuation.json'
                    calls = [meta.get('call_id', '')]
                    if continuation.is_file():
                        calls.append(json.loads(continuation.read_text()).get('call_id', ''))
                    for call_id in calls:
                        if call_id:
                            cancel(str(call_id))
                    shutil.rmtree(folder, ignore_errors=True)
        volume.commit()

    def backend() -> CloudVideoUploads:
        return CloudVideoUploads(STORE, lambda job_id: capture_job.spawn(job_id).object_id,
                                 poll, cancel, volume.reload, volume.commit, limits)

    return backend, volume
