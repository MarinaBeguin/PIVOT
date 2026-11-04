"""The Modal adapter's worker chain is exercised with local fake SDK objects."""
import importlib.util
import json
import sys
from collections.abc import Callable
from typing import Any
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pivot.scene.examples import office_scene


class Image:
    """Fluent image builder that never installs anything."""
    def __getattr__(self, _name) -> Callable[..., Any]:
        return lambda *_args, **_kwargs: self


class Volume:
    """Local no-op volume refresh/commit callbacks."""
    @classmethod
    def from_name(cls, *_args, **_kwargs) -> "Volume":
        return cls()

    def reload(self):
        pass

    def commit(self):
        pass


class App:
    """Records workers and executes spawned test jobs synchronously."""
    def __init__(self):
        self.functions = {}
        self.results = {}

    def function(self, **options) -> Callable[..., Any]:
        def decorate(function) -> SimpleNamespace:
            def spawn(*args) -> SimpleNamespace:
                result = function(*args)
                call_id = 'call-' + options['name']
                self.results[call_id] = result
                return SimpleNamespace(object_id=call_id)
            wrapper = SimpleNamespace(spawn=spawn, raw=function, options=options)
            self.functions[options['name']] = wrapper
            return wrapper
        return decorate


def test_modal_gpu_to_cpu_chain_and_expiry_cleanup(tmp_path, monkeypatch):
    app = App()
    fake = SimpleNamespace(Volume=Volume, Image=SimpleNamespace(debian_slim=lambda **_: Image()),
                           Cron=lambda spec: spec)
    monkeypatch.setitem(sys.modules, 'modal', fake)
    path = Path(__file__).resolve().parents[1] / 'deploy/modal_video.py'
    spec = importlib.util.spec_from_file_location('test_modal_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.STORE = tmp_path / 'jobs'
    monkeypatch.setattr('pivot.server.cloud_uploads.validate_video', lambda *_: None)
    commands = []

    def run(command, *, timeout, env=None):
        commands.append((command, timeout, env))
        if 'pivot.reconstruction.video_capture' in command:
            folder = Path(command[command.index('--out') + 1])
            folder.mkdir(parents=True)
            (folder / 'video.mp4').touch()
        elif command[0] == 'bash':
            scene = Path(command[-1]) / 'scene'
            scene.mkdir(parents=True)
            office_scene().save(scene / 'scene.json')
            np.savez(scene / 'cloud.npz', points=np.zeros((1, 3), np.float32),
                     colors=np.zeros((1, 3), np.uint8))

    monkeypatch.setattr(module, "_run", run)
    factory, _ = module.attach_uploads(app, Image(), None)
    uploads = factory()
    source = tmp_path / 'video.mov'
    source.write_bytes(b'fake video')
    job = uploads.start_file(source, 'video.mov', 'browser')
    assert [c[0][0] for c in commands] == [sys.executable, sys.executable, 'bash']
    assert commands[-1][2]['PIVOT_MAX_OBJECTS_API'] == '20'
    assert commands[-1][2]['PIVOT_REQUIRE_INVENTORY'] == '1'
    assert commands[-1][2]['PIVOT_VIDEO_STRIDE'] == '1'
    assert commands[-1][2]['PIVOT_NO_SERVE'] == '1'
    assert uploads.status(job, 'browser')['state'] == 'ready'
    assert app.functions['video_capture'].options['gpu'] == 'L40S'
    assert app.functions['video_capture'].options['max_containers'] == 1
    assert app.functions['video_scene'].options['retries'] == 0
    meta = uploads.folder / job / 'meta.json'
    value = json.loads(meta.read_text())
    meta.write_text(json.dumps({**value, 'expires': 0}))
    app.functions['video_cleanup'].raw()
    assert not meta.parent.exists()
