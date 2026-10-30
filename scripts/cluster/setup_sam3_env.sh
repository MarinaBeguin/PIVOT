#!/usr/bin/env bash
# Creates the GPU environment for self-hosted SAM 3.1. Run once on hades:
#   bash scripts/cluster/setup_sam3_env.sh
# Prerequisites: access granted to https://huggingface.co/facebook/sam3.1
# and a Hugging Face token (the script runs `hf auth login` if none is stored).
set -euo pipefail

SCRATCH="${SCRATCH:-/itet-stor/mbeguin/net_scratch}"
CONDA="${CONDA_ROOT:-$SCRATCH/conda}/bin/conda"
ENV="$SCRATCH/conda_envs/pivot_sam3"
SAM3_SRC="$SCRATCH/external/sam3"
PIVOT_SRC="$(cd "$(dirname "$0")/../.." && pwd)"
# /tmp on hades is a 1 GB tmpfs, too small to unpack the CUDA wheels.
export TMPDIR="$SCRATCH/tmp" PIP_CACHE_DIR="$SCRATCH/pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [ ! -x "$ENV/bin/python" ]; then
  "$CONDA" create -y -p "$ENV" python=3.12
fi
PIP="$ENV/bin/python -m pip"
$PIP install --upgrade pip
# torch 2.8 + CUDA 12.6 is the combination already known to work on this cluster.
$PIP install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126

if [ ! -d "$SAM3_SRC" ]; then
  git clone https://github.com/facebookresearch/sam3.git "$SAM3_SRC"
else
  git -C "$SAM3_SRC" pull --ff-only
fi
# sam3 still imports pkg_resources, which setuptools 81 removed.
$PIP install "setuptools<81"
$PIP install -e "$SAM3_SRC"
# Inference dependencies sam3 does not declare (it lists them under "notebooks").
$PIP install einops opencv-python-headless psutil pycocotools scikit-image scikit-learn \
  matplotlib "numpy>=1.26,<2"
# sam3 pins numpy<2; the constraint makes pip pick a matching scipy for PIVOT.
$PIP install -e "$PIVOT_SRC" "numpy>=1.26,<2"
# hades has no ffmpeg; this one encodes the faster, subsampled videos (--quality).
$PIP install imageio-ffmpeg
# Spoken requests in the open version: wav2vec 2.0 on the server's CPU.
$PIP install "transformers>=4.45,<5" "numpy>=1.26,<2"
$PIP check

if ! "$ENV/bin/hf" auth whoami >/dev/null 2>&1; then
  read -rsp "Hugging Face read token: " HF_TOKEN && echo
  "$ENV/bin/hf" auth login --token "$HF_TOKEN"
  unset HF_TOKEN
fi
"$ENV/bin/python" -c "from sam3.model_builder import build_sam3_multiplex_video_predictor; \
import torch, numpy, pivot; print('torch', torch.__version__, 'numpy', numpy.__version__, 'ok')"
# Download the checkpoints here so GPU jobs start without network access to Hugging Face.
HF_HUB_CACHE="$SCRATCH/hf_cache" "$ENV/bin/hf" download facebook/sam3.1
HF_HUB_CACHE="$SCRATCH/hf_cache" "$ENV/bin/hf" download facebook/wav2vec2-large-960h-lv60-self
