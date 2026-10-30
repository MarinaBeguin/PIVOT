#!/usr/bin/env bash
# PIVOT, open-weight version: no API key needed. Llama runs on our own GPUs
# (Slurm jobs); prerequisites: scripts/cluster/setup_sam3_env.sh and
# scripts/cluster/setup_llama_env.sh, run once.
#
#   bash scripts/run_open.sh [--quality fast|medium|full] <scannetpp_scene_dir> <video.mp4> \
#       <out_dir> [prompt ...]
#
# Objects come from the best source available:
#   1. SAM 3.1 API results already computed for this video: reused, no key needed;
#   2. else, with an API key: Muse Spark and SAM 3.1 through the API;
#   3. else: Llama 3.2 Vision lists the objects (up to PIVOT_MAX_OBJECTS_OPEN, 8)
#      and self-hosted SAM 3.1 tracks them (PIVOT_SAM_WORKERS GPU jobs, 3).
# Then builds the scene, starts Llama on a GPU and serves the viewer with the
# agent. Jobs are cancelled on exit (Ctrl-C). Steps already done are skipped.
set -euo pipefail
source "$(dirname "$0")/pipeline_common.sh"
export PIVOT_BACKEND=open

DETECTIONS=$(detections_dir sam_api)
if prompts_done_in "$DETECTIONS"; then
  echo "== [1/4] Reusing the SAM 3.1 API results of this video (no key needed)"
  choose_prompts false
  echo "== [2/4] Objects already recognized"
elif has_api_key; then
  echo "== [1/4] Listing the objects with Muse Spark (API key found: better objects)"
  choose_prompts list_objects_api
  echo "== [2/4] Recognizing objects with SAM 3.1 through the Meta API ($QUALITY quality)"
  segment_api
else
  DETECTIONS=$(detections_dir sam_local)
  echo "== [1/4] Listing the objects of the video with Llama 3.2 Vision"
  choose_prompts list_objects_local
  echo "== [2/4] Recognizing objects with self-hosted SAM 3.1 ($QUALITY quality)"
  segment_local
fi

echo "== [3/4] Building the 3D scene"
# Built before Llama starts, so that no GPU waits during the build.
build_scene

echo "== [4/4] Starting Llama (GPU job)"
JOBS=("$(sbatch --parsable "$PIVOT_SRC/scripts/cluster/llama_server.sbatch")")
LOG="$LOGS/llama_${JOBS[0]}.out"
until [ -f "$LOG" ] && grep -q "OpenAI-compatible API at" "$LOG"; do sleep 10; done
PIVOT_OPEN_URL=$(grep -o "http://[^ ]*/v1" "$LOG" | head -1)
export PIVOT_OPEN_URL
echo "   waiting for the model to load at $PIVOT_OPEN_URL"
until curl -sf "$PIVOT_OPEN_URL/models" >/dev/null; do sleep 15; done
serve
