#!/usr/bin/env bash
# Lists the kinds of objects a room video shows, with Llama 3.2 Vision (open
# weights), to choose SAM 3.1's prompts. Called by scripts/run_open.sh.
#
#   export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf
#   sbatch scripts/cluster/inventory.sbatch <video.mp4> <prompts.txt> [max_objects]
#
# Prerequisite: scripts/cluster/setup_llama_env.sh (environment and weights).
#SBATCH --job-name=pivot-inventory
#SBATCH --partition=gpu.all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/itet-stor/mbeguin/net_scratch/pivot/logs/inventory_%j.out
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: sbatch $0 <video.mp4> <prompts.txt> [max_objects]" >&2
  exit 2
fi
ENV="${PIVOT_LLAMA_ENV:-/itet-stor/mbeguin/net_scratch/conda_envs/pivot_llama}"
# Slurm runs a copy of this script: the sources are found from where it was submitted.
SRC="${PIVOT_SRC:-$SLURM_SUBMIT_DIR}"
export PYTHONPATH="$SRC/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_CACHE="${HF_HUB_CACHE:-/itet-stor/mbeguin/net_scratch/hf_cache}" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
exec "$ENV/bin/python" -m pivot.perception.inventory --backend open --video "$1" --out "$2" \
  --max-objects "${3:-8}"
