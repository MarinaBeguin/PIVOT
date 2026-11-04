"""Remote video admission, recovery and browser isolation without Modal/GPU/API calls."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pivot.scene.examples import office_scene
from pivot.server.app import create_app
from pivot.server.limits import Budget, Meter, SpendLedger
from pivot.server.cloud_uploads import CloudVideoUploads, UploadError, write_state
from test_server import office


@pytest.fixture(name="uploads")
def _uploads(tmp_path, monkeypatch) -> CloudVideoUploads:
    monkeypatch.setattr('pivot.server.cloud_uploads.validate_video', lambda *_: None)
    return CloudVideoUploads(tmp_path / 'jobs', lambda _: 'remote-call', lambda _: 'running',
                             lambda _: None)


def source(tmp_path: Path) -> Path:
    path = tmp_path / 'input.mp4'
    path.write_bytes(b'video')
    return path


def finish(uploads, job):
    folder = uploads.folder / job
    scene = folder / 'out/room/scene'
    scene.mkdir(parents=True)
    office_scene().save(scene / 'scene.json')
    np.savez(scene / 'cloud.npz', points=np.zeros((3, 3), np.float32),
             colors=np.zeros((3, 3), np.uint8))
    write_state(folder, 'ready', 'Ready')


def client(uploads) -> TestClient:
    return TestClient(create_app(office(np.zeros((3, 3)), np.zeros((3, 3), dtype=np.uint8)),
                                 uploads=uploads))


def test_quota_is_persistent_and_deletion_does_not_refund_it(uploads, tmp_path):
    file = source(tmp_path)
    job = uploads.start_file(file, 'my.MOV', 'owner', 'ip')
    assert uploads.owns(job, 'owner') and not uploads.owns(job, 'other')
    assert not uploads.owns('../usage.json', 'owner')
    resumed = replace(uploads)
    assert resumed.status(job, 'owner')['state'] == 'running'
    assert resumed.delete(job, 'owner')
    assert resumed.status(job, 'owner') is None
    with pytest.raises(UploadError, match='quota'):
        resumed.start_file(file, 'my.MOV', 'new-cookie', 'ip')


def test_active_job_cap_prevents_an_unbounded_gpu_queue(uploads, tmp_path):
    file = source(tmp_path)
    uploads.start_file(file, 'video.mp4', 'a', 'ip-a')
    with pytest.raises(UploadError, match='progress'):
        uploads.start_file(file, 'video.mp4', 'b', 'ip-b')


def test_remote_failure_is_safe_and_does_not_publish_room(uploads, tmp_path):
    job = uploads.start_file(source(tmp_path), 'video.mp4', 'a')
    uploads.poll = lambda _: 'failed'
    status = uploads.status(job, 'a')
    assert status['state'] == 'failed' and 'room' not in status
    assert 'remote-call' not in json.dumps(status)


def test_expired_job_is_not_accessible_even_before_cleanup(uploads, tmp_path):
    job = uploads.start_file(source(tmp_path), 'video.mp4', 'a')
    meta = uploads.folder / job / 'meta.json'
    data = json.loads(meta.read_text())
    meta.write_text(json.dumps({**data, 'expires': 0}))
    assert not uploads.owns(job, 'a')
    assert uploads.status(job, 'a') is None


def test_browser_owner_can_recover_room_after_web_restart_but_other_browser_cannot(uploads):
    first = client(uploads)
    other = TestClient(first.app)
    response = first.post('/api/video', content=b'video', headers={'X-Filename': 'room.mov'})
    job = response.json()['job']
    finish(uploads, job)
    assert first.get('/api/video/' + job).json()['state'] == 'ready'
    assert other.get('/api/video/' + job).status_code == 404
    assert other.get('/?scene=' + job).status_code == 404
    assert other.get('/api/objects?scene=' + job).status_code == 404
    assert other.delete('/api/video/' + job).status_code == 404
    cookie = first.cookies.get('pivot_video_owner')
    restored = client(replace(uploads))
    restored.cookies.set('pivot_video_owner', cookie)
    assert restored.get('/?scene=' + job).status_code == 200
    assert restored.get('/api/objects?scene=' + job).status_code == 200
    assert restored.delete('/api/video/' + job).json()['deleted']
    assert restored.get('/?scene=' + job).status_code == 404


def test_upload_size_is_checked_on_actual_stream_not_just_header(uploads):
    uploads.limits = replace(uploads.limits, max_bytes=3)
    response = client(uploads).post('/api/video', content=b'1234')
    assert response.status_code == 413
    assert not uploads.folder.exists()


def test_ready_without_cloud_file_is_not_reported_as_ready(uploads, tmp_path):
    job = uploads.start_file(source(tmp_path), 'video.mp4', 'a')
    write_state(uploads.folder / job, 'ready', 'Ready')
    assert uploads.status(job, 'a')['state'] == 'failed'


def test_public_visitor_can_upload_without_the_owner_key(uploads):
    meter = Meter(SpendLedger(Budget()), 'salt', proxy_hops=0, owner_key='owner-only')
    app = TestClient(create_app(office(np.zeros((3, 3)), np.zeros((3, 3), dtype=np.uint8)),
                                meter=meter, uploads=uploads))
    response = app.post('/api/video', content=b'video')
    assert response.status_code == 200 and response.json()['job'].startswith('video-')


def test_invalid_media_does_not_reserve_quota_or_start_a_job(uploads, monkeypatch):
    seen = []

    def reject(path, _duration):
        seen.append(path)
        raise UploadError('Invalid video')

    monkeypatch.setattr('pivot.server.cloud_uploads.validate_video', reject)
    response = client(uploads).post('/api/video', content=b'invalid media')
    assert response.status_code == 422
    assert seen and not seen[0].exists()
    assert not uploads.folder.exists()
