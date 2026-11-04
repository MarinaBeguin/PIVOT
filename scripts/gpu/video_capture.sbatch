#!/usr/bin/env bash
# Gives a phone video its cameras and metric depth with MapAnything, as a capture.
#
#   sbatch scripts/gpu/video_capture.sbatch <video> <capture_dir> [frames]
#
#SBATCH --job-name=pivot-capture
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: sbatch $0 <video> <capture_dir> [frames]" >&2
  exit 2
fi
ENV="${PIVOT_MAPANYTHING_ENV:-${PIVOT_SCRATCH:-$HOME/pivot_scratch}/conda_envs/pivot_mapanything}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_CACHE="${HF_HUB_CACHE:-${PIVOT_SCRATCH:-$HOME/pivot_scratch}/hf_cache}"
export PYTHONPATH="${PIVOT_SRC:?}/src${PYTHONPATH:+:$PYTHONPATH}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$ENV/bin/python" -m pivot.reconstruction.video_capture --video "$1" --out "$2" \
  --frames "${3:-150}"
