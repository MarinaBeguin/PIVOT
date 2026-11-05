#!/usr/bin/env bash
# CPU-only contracts, evaluated separately from the live-model benchmark.
set -euo pipefail
EVAL_ROOT="$(cd "$(dirname "$0")" && pwd)"
CODE_ROOT="${PIVOT_EVAL_CODE_ROOT:-$(pwd)}"
PYTHON="${PIVOT_PYTHON:-/itet-stor/mbeguin/net_scratch/conda_envs/pivot_sam3/bin/python}"
OUT="${1:?Usage: bash eval/run_safety.sh OUTPUT_DIRECTORY}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
export PYTHONPATH="$CODE_ROOT/src:$CODE_ROOT/tests"
"$PYTHON" -m pytest -o addopts='' -p no:cacheprovider -q \
  "$EVAL_ROOT/safety/test_dialogue.py" \
  "$CODE_ROOT/tests/test_access_routes.py" \
  "$CODE_ROOT/tests/test_access_safety.py" \
  "$CODE_ROOT/tests/test_empty_answer.py" \
  "$CODE_ROOT/tests/test_motion.py::test_budget_exhaustion_is_not_reported_as_impossible" \
  --junitxml="$OUT/safety.xml" 2>&1 | tee "$OUT/safety.log"
