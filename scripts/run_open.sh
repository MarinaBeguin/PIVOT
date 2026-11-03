#!/usr/bin/env bash
# PIVOT, open-weight version: no API key needed. Llama runs on our own GPUs
# (Slurm jobs); prerequisites: scripts/gpu/setup_sam3_env.sh and
# scripts/gpu/setup_llama_env.sh, run once.
#
#   bash scripts/run_open.sh [--quality fast|medium|full] <scannetpp_scene_dir> <video.mp4> \
#       <out_dir> [prompt ...]
#
# Nothing is sent to the Meta API, even with a key in .env. Objects come from:
#   1. SAM 3.1 API results already computed for this video (by run_api.sh):
#      reused as they are, nothing is requested;
#   2. else: Llama 3.2 Vision lists the objects (up to PIVOT_MAX_OBJECTS_OPEN, 8)
#      and self-hosted SAM 3.1 tracks them (PIVOT_SAM_WORKERS GPU jobs, 3).
# Then builds the scene, starts the agent's model on a GPU (Llama 3.1 8B, or
# Muse Glimmer with PIVOT_OPEN_AGENT=glimmer) and serves the viewer with it.
# Jobs are cancelled on exit (Ctrl-C). Steps already done are skipped.
set -euo pipefail
source "$(dirname "$0")/pipeline_common.sh"
export PIVOT_BACKEND=open

DETECTIONS=$(detections_dir sam_api)
# Results computed on every frame, before keyframes, count too.
if ! prompts_done_in "$DETECTIONS" && prompts_done_in "$(KEYFRAMES=0 detections_dir sam_api)"; then
  DETECTIONS=$(KEYFRAMES=0 detections_dir sam_api)
fi
if prompts_done_in "$DETECTIONS"; then
  echo "== [1/4] Reusing the SAM 3.1 API results of this video (no key needed)"
  choose_prompts false
  echo "== [2/4] Objects already recognized"
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

if [ "${PIVOT_OPEN_AGENT:-llama}" = glimmer ]; then
  # Muse Glimmer, distilled from Muse Spark: two 48 GB GPUs for its bf16 weights.
  export PIVOT_AGENT_MODEL=meta-models/Muse-Glimmer-30B
  echo "== [4/4] Starting Muse Glimmer (GPU job, 2 GPUs)"
  JOBS=("$(sbatch --parsable --output="$LOGS/llama_%j.out" --gres=gpu:2 --mem=96G \
    ${PIVOT_SLURM_PARTITION:+--partition="$PIVOT_SLURM_PARTITION"} \
    "$PIVOT_SRC/scripts/gpu/llama_server.sbatch" "$PIVOT_AGENT_MODEL")")
else
  echo "== [4/4] Starting Llama (GPU job)"
  JOBS=("$(submit llama llama_server)")
fi
LOG="$LOGS/llama_${JOBS[0]}.out"
until [ -f "$LOG" ] && grep -q "OpenAI-compatible API at" "$LOG"; do sleep 10; done
PIVOT_OPEN_URL=$(grep -o "http://[^ ]*/v1" "$LOG" | head -1)
export PIVOT_OPEN_URL
echo "   waiting for the model to load at $PIVOT_OPEN_URL"
until curl -sf "$PIVOT_OPEN_URL/models" >/dev/null; do sleep 15; done
serve
