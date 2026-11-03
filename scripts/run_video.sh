#!/usr/bin/env bash
# PIVOT on your own video: any phone (iPhone or Android), no LiDAR, no app.
#
#   bash scripts/run_video.sh <video> <out_dir> [prompt ...]
#
# 1. MapAnything (GPU job) estimates each frame's camera and metric depth, and writes a
#    capture in the layout of a ScanNet++ one (<out_dir>/capture);
# 2. SAM 3.1 finds people, faces, plates and writing, which are blurred in the frames
#    before anything else sees them (PIVOT_PRIVACY=0 to skip; through the Meta API);
# 3. then the API version runs on the capture: Muse Spark lists the objects, SAM 3.1
#    tracks them, the scene is built and served with the agent (see run_api.sh).
# Each step is skipped when already done. PIVOT_CAPTURE_FRAMES sets the frames kept (150).
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo "usage: $0 <video> <out_dir> [prompt ...]" >&2
  exit 2
fi
VIDEO_IN="$(realpath "$1")"; OUT_DIR="$(realpath -m "$2")"; shift 2
export PIVOT_SRC="$(cd "$(dirname "$0")/.." && pwd)"
[ ! -f "$PIVOT_SRC/scripts/site.env" ] || source "$PIVOT_SRC/scripts/site.env"
export PYTHONPATH="$PIVOT_SRC/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PIVOT_PYTHON:-${PIVOT_SCRATCH:-$HOME/pivot_scratch}/conda_envs/pivot_sam3/bin/python}"
CAPTURE="$OUT_DIR/capture"
mkdir -p "$OUT_DIR"

echo "== [A] Cameras and depth of every frame (MapAnything, GPU job)"
if [ ! -f "$CAPTURE/video.mp4" ]; then
  JOB=$(sbatch --parsable ${PIVOT_SLURM_PARTITION:+--partition="$PIVOT_SLURM_PARTITION"} \
    --output="$OUT_DIR/capture_%j.out" --export=ALL \
    "$PIVOT_SRC/scripts/gpu/video_capture.sbatch" "$VIDEO_IN" "$CAPTURE" \
    "${PIVOT_CAPTURE_FRAMES:-150}")
  echo "   job $JOB; log: $OUT_DIR/capture_$JOB.out"
  while squeue -h -j "$JOB" 2>/dev/null | grep -q .; do sleep 15; done
  [ -f "$CAPTURE/video.mp4" ] || { echo "   reconstruction failed, see the log" >&2; exit 1; }
else
  echo "   already done: $CAPTURE"
fi

if [ "${PIVOT_PRIVACY:-1}" = 1 ]; then
  echo "== [B] Blurring people, faces, plates and writing (SAM 3.1)"
  (cd "$PIVOT_SRC" && "$PYTHON" -m pivot.privacy --capture "$CAPTURE" \
    --env-file "$PIVOT_SRC/.env")
fi

echo "== [C] PIVOT on the capture"
# Frame i of the capture's video is image i of the capture.
PIVOT_VIDEO_STRIDE=1 exec bash "$PIVOT_SRC/scripts/run_api.sh" \
  "$CAPTURE" "$CAPTURE/video.mp4" "$OUT_DIR/room" "$@"
