#!/usr/bin/env bash
set -euo pipefail
exec python -m pivot.server --scene-dir "${PIVOT_SCENE_DIR:-/scene}" \
  --host 0.0.0.0 --port "${PORT:-8000}" --chat "$@"
