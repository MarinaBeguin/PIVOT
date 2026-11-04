#!/usr/bin/env bash
# Local equivalents of the cluster GPU jobs. Weights remain in the mounted cache.
set -euo pipefail
command="${1:-help}"
[ "$#" -eq 0 ] || shift
case "$command" in
  video) exec bash /app/scripts/run_video.sh "$@" ;;
  capture) exec "${PIVOT_CAPTURE_PYTHON:-python}" -m pivot.reconstruction.video_capture "$@" ;;
  inventory) exec python -m pivot.perception.inventory --backend open "$@" ;;
  segment) exec python -m pivot.perception.sam3_local "$@" ;;
  privacy) exec python -m pivot.privacy "$@" ;;
  build) exec python -m pivot.build "$@" ;;
  agent) exec python -m pivot.agent "$@" ;;
  eval) exec python -m pivot.eval "$@" ;;
  layout) exec python -m pivot.lifting.scenescript "$@" ;;
  serve) exec bash /app/docker/serve.sh "$@" ;;
  fetch-models)
    [ "$#" -gt 0 ] || { echo 'Supply Hugging Face repository ids' >&2; exit 2; }
    exec python -c 'import sys; from huggingface_hub import snapshot_download; [snapshot_download(repo_id=name) for name in sys.argv[1:]]' "$@"
    ;;
  check)
    ffmpeg -version | head -1
    python - <<'PY'
import importlib
import torch
for name in ('pivot', 'sam3', 'transformers', 'cv2'):
    importlib.import_module(name)
    print(name, 'import OK')
print('torch', torch.__version__, 'CUDA build', torch.version.cuda)
x = torch.ones(1, device='cuda')
print('GPU', torch.cuda.get_device_name(), 'result', (x + 1).item())
PY
    "${PIVOT_CAPTURE_PYTHON:-python}" -c 'import mapanything, numpy, torch; print("MapAnything", numpy.__version__, torch.__version__); print(torch.ones(1, device="cuda"))'
    ;;
  *) echo 'Usage: bash docker/tools.sh {check|fetch-models|video|capture|inventory|segment|privacy|build|agent|eval|layout|serve} [arguments...]' >&2; exit 2 ;;
esac
