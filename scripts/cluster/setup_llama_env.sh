#!/usr/bin/env bash
# Creates the environment serving Llama with vLLM, for the open-weight version of
# PIVOT (no API key: SAM 3.1 and Llama both run on our GPUs). Run once on hades:
#   bash scripts/cluster/setup_llama_env.sh [model]
# Prerequisites: access granted to both models on Hugging Face (meta-llama models
# are gated, each on its own page) and a Hugging Face token stored by `hf auth login`.
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
# Names the objects of a video (scripts/cluster/inventory.sbatch); gated separately.
VISION_MODEL="${PIVOT_VISION_MODEL:-meta-llama/Llama-3.2-11B-Vision-Instruct}"
SCRATCH="${SCRATCH:-/itet-stor/mbeguin/net_scratch}"
CONDA="${CONDA_ROOT:-$SCRATCH/conda}/bin/conda"
ENV="$SCRATCH/conda_envs/pivot_llama"
# /tmp on hades is a 1 GB tmpfs, too small to unpack the CUDA wheels.
export TMPDIR="$SCRATCH/tmp" PIP_CACHE_DIR="$SCRATCH/pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [ ! -x "$ENV/bin/python" ]; then
  "$CONDA" create -y -p "$ENV" python=3.12
fi
# vLLM brings its own torch build; it gets an environment of its own.
"$ENV/bin/python" -m pip install --upgrade pip
"$ENV/bin/python" -m pip install "vllm>=0.10,<0.12"
"$ENV/bin/python" -c "import vllm; print('vllm', vllm.__version__)"
# Download the weights here so GPU jobs start without network access to Hugging Face.
for model in "$MODEL" "$VISION_MODEL"; do
  HF_HUB_CACHE="$SCRATCH/hf_cache" "$ENV/bin/hf" download "$model"
done
