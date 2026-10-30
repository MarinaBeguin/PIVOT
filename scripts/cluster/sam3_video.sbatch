#!/usr/bin/env bash
# Tracks text prompts through a video with self-hosted SAM 3.1.
#
#   export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf
#   sbatch scripts/cluster/sam3_video.sbatch <video.mp4> <out_dir> <prompt> [<prompt> ...]
#
# Prompts with spaces must be quoted, e.g. "trash can".
#SBATCH --job-name=pivot-sam3
#SBATCH --partition=gpu.all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/itet-stor/mbeguin/net_scratch/pivot/logs/sam3_%j.out
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: sbatch $0 <video.mp4> <out_dir> <prompt> [<prompt> ...]" >&2
  exit 2
fi
VIDEO="$1"; OUT="$2"; shift 2

ENV="${PIVOT_SAM3_ENV:-/itet-stor/mbeguin/net_scratch/conda_envs/pivot_sam3}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Limits fragmentation when mask sizes vary from frame to frame.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The token stays in ~/.cache/huggingface; the checkpoints go to scratch storage.
export HF_HUB_CACHE="${HF_HUB_CACHE:-/itet-stor/mbeguin/net_scratch/hf_cache}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$ENV/bin/python" -m pivot.perception.sam3_local --video "$VIDEO" --out "$OUT" --prompts "$@"
