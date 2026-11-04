"""The local video runner must not submit cluster jobs."""
import os
from pathlib import Path
import shutil
import subprocess


def test_local_video_runs_capture_privacy_pipeline_without_slurm(tmp_path: Path):
    root = tmp_path / 'checkout'
    scripts = root / 'scripts'
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1]
    shutil.copy(source / 'scripts/run_video.sh', scripts)
    log = tmp_path / 'calls'
    python = tmp_path / 'fake-python'
    python.write_text('''#!/usr/bin/env bash
set -eu
echo "$*" >> "$CALL_LOG"
if [ "$2" = pivot.reconstruction.video_capture ]; then
  while [ "$1" != --out ]; do shift; done
  mkdir -p "$2"; touch "$2/video.mp4"
fi
''')
    python.chmod(0o755)
    (scripts / 'run_api.sh').write_text('echo "api stride=$PIVOT_VIDEO_STRIDE $*" >> "$CALL_LOG"\n')
    video = tmp_path / 'my video.MOV'
    video.touch()
    env = {**os.environ, 'PIVOT_EXECUTOR': 'local', 'PIVOT_PYTHON': str(python),
           'PIVOT_CAPTURE_PYTHON': str(python), 'PIVOT_CAPTURE_FRAMES': '8',
           'PIVOT_PRIVACY': '1', 'CALL_LOG': str(log)}
    command = ['bash', str(scripts / 'run_video.sh'), str(video), str(tmp_path / 'out')]
    subprocess.run(command, env=env, check=True, capture_output=True)
    calls = log.read_text()
    assert '--frames 8' in calls and 'pivot.privacy' in calls and 'api stride=1' in calls
    log.write_text('')
    subprocess.run(command, env=env, check=True, capture_output=True)
    assert 'pivot.reconstruction.video_capture' not in log.read_text()
