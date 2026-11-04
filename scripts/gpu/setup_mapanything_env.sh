#!/usr/bin/env bash
# Creates the GPU environment for MapAnything, which gives any phone video its cameras
# and metric depth (see pivot.reconstruction.video_capture). Run once on the machine
# that submits the GPU jobs:
#   bash scripts/gpu/setup_mapanything_env.sh
# The Apache 2.0 checkpoint (facebook/map-anything-apache) downloads on first use.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Settings of this machine (PIVOT_SCRATCH, CONDA_ROOT...), if any; never committed.
[ ! -f "$ROOT/scripts/site.env" ] || source "$ROOT/scripts/site.env"
SCRATCH="${PIVOT_SCRATCH:-$HOME/pivot_scratch}"
CONDA="${CONDA_ROOT:-$SCRATCH/conda}/bin/conda"
ENV="$SCRATCH/conda_envs/pivot_mapanything"
SOURCE="$SCRATCH/external/map-anything"
PIVOT_SRC="$(cd "$(dirname "$0")/../.." && pwd)"
# /tmp can be a small tmpfs, too small to unpack the CUDA wheels.
export TMPDIR="$SCRATCH/tmp" PIP_CACHE_DIR="$SCRATCH/pip_cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [ ! -x "$ENV/bin/python" ]; then
  "$CONDA" create -y -p "$ENV" python=3.12
fi
PIP="$ENV/bin/python -m pip"
$PIP install --upgrade pip
$PIP install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
if [ ! -d "$SOURCE" ]; then
  git clone https://github.com/facebookresearch/map-anything.git "$SOURCE"
else
  git -C "$SOURCE" pull --ff-only
fi
$PIP install -e "$SOURCE"
$PIP install -e "$PIVOT_SRC" imageio-ffmpeg
