#!/usr/bin/env bash
# Held-out evaluation of PIVOT: requests on rooms never used while PIVOT was developed.
#
#   bash eval/run_heldout.sh [variant ...]
#
# For each room (bedroom, classroom, apartment): makes the requests if eval/heldout/<room>/
# has none (geometry only, no API), runs the variants (default: all four; the model ones
# need META_API_KEY in .env), and writes results.jsonl, summary.md and summary.json.
# An interrupted run resumes where it stopped. Scenes are read from $PIVOT_RUNS/<room>/scene
# (directories written by pivot.build), $PIVOT_SCRATCH/pivot/runs by default.
#
# With open weights, no API key: PIVOT_EVAL_OPEN=llama (Llama 3.1 8B, one GPU) or glimmer
# (Muse Glimmer 30B, two GPUs) starts the model as a GPU job, runs the given variants
# (default: llm_only and pivot) with it, recorded as <variant>_llama or <variant>_glimmer.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ ! -f "$ROOT/scripts/site.env" ] || source "$ROOT/scripts/site.env"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PIVOT_PYTHON:-${PIVOT_SCRATCH:-$HOME/pivot_scratch}/conda_envs/pivot_sam3/bin/python}"
RUNS="${PIVOT_RUNS:-${PIVOT_SCRATCH:-$HOME/pivot_scratch}/pivot/runs}"
VARIANTS=("$@")
OPEN="${PIVOT_EVAL_OPEN:-}"
TAG=()
if [ -n "$OPEN" ]; then
  [ "${#VARIANTS[@]}" -gt 0 ] || VARIANTS=(llm_only pivot)
  LOGS="${PIVOT_LOGS:-${PIVOT_SCRATCH:-$HOME/pivot_scratch}/pivot/logs}"
  mkdir -p "$LOGS"
  case "$OPEN" in
    llama) MODEL=meta-llama/Llama-3.1-8B-Instruct; GPUS=1 ;;
    glimmer) MODEL=meta-models/Muse-Glimmer-30B; GPUS=2 ;;
    *) echo "PIVOT_EVAL_OPEN must be llama or glimmer" >&2; exit 2 ;;
  esac
  JOB=$(sbatch --parsable --output="$LOGS/eval_${OPEN}_%j.out" --gres=gpu:$GPUS \
    --mem=$((48 * GPUS))G ${PIVOT_SLURM_PARTITION:+--partition="$PIVOT_SLURM_PARTITION"} \
    "$ROOT/scripts/gpu/llama_server.sbatch" "$MODEL")
  trap 'scancel "$JOB"' EXIT
  LOG="$LOGS/eval_${OPEN}_$JOB.out"
  echo "== $MODEL: GPU job $JOB, log $LOG"
  until [ -f "$LOG" ] && grep -q "OpenAI-compatible API at" "$LOG"; do sleep 10; done
  PIVOT_OPEN_URL=$(grep -o "http://[^ ]*/v1" "$LOG" | head -1)
  until curl -sf "$PIVOT_OPEN_URL/models" >/dev/null; do sleep 15; done
  export PIVOT_BACKEND=open PIVOT_OPEN_URL PIVOT_AGENT_MODEL="$MODEL"
  TAG=(--tag "_$OPEN")
fi
[ "${#VARIANTS[@]}" -gt 0 ] || VARIANTS=(planner_only llm_only llm_check pivot)
cd "$ROOT"
for room in bedroom classroom apartment; do
  out="eval/heldout/$room"
  mkdir -p "$out"
  echo "== $room"
  if [ ! -s "$out/tasks.json" ]; then
    "$PYTHON" -m pivot.eval generate --scene "$RUNS/$room/scene" \
      --tasks "$out/tasks.json" --count 44 --seed 1
    sha256sum "$RUNS/$room/scene/scene.json" | cut -d' ' -f1 > "$out/scene.sha256"
  fi
  # The requests name objects by number: they only hold for the scene they were made on.
  if ! sha256sum "$RUNS/$room/scene/scene.json" | grep -q "^$(cat "$out/scene.sha256") "; then
    echo "$RUNS/$room/scene is not the scene these requests were made on" \
      "(remove $out/tasks.json to make new ones)" >&2
    exit 1
  fi
  "$PYTHON" -m pivot.eval run --scene "$RUNS/$room/scene" --tasks "$out/tasks.json" \
    --out "$out/results.jsonl" --variants "${VARIANTS[@]}" "${TAG[@]}" --env-file "$ROOT/.env" > /dev/null
  "$PYTHON" -m pivot.eval summary --out "$out/results.jsonl" --json "$out/summary.json" \
    > "$out/summary.md"
  cat "$out/summary.md"
done
