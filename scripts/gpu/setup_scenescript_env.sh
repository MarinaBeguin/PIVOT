#!/usr/bin/env bash
# Sets up Meta's SceneScript (CC BY-NC) once: its code and its conda environment.
#
#   bash scripts/gpu/setup_scenescript_env.sh
#
# The weights are not downloaded here: request them on SceneScript's page
# (https://www.projectaria.com/scenescript/), then put the checkpoint (.ckpt) at
# $PIVOT_SCRATCH/scenescript/scenescript_model_ase.ckpt. Building the environment
# compiles torchsparse and takes up to 30 minutes; it needs CUDA.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[ ! -f "$ROOT/scripts/site.env" ] || source "$ROOT/scripts/site.env"
SCRATCH="${PIVOT_SCRATCH:-$HOME/pivot_scratch}"
DIR="$SCRATCH/scenescript"
ENV="$SCRATCH/conda_envs/scenescript"
mkdir -p "$DIR"
[ -d "$DIR/code" ] || git clone https://github.com/facebookresearch/scenescript.git "$DIR/code"
if [ ! -x "$ENV/bin/python" ]; then
  conda env create --file "$DIR/code/environment.yaml" --prefix "$ENV"
fi
"$ENV/bin/python" -c "import torch, torchsparse; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
[ -f "$DIR/scenescript_model_ase.ckpt" ] || echo "Now put the checkpoint at $DIR/scenescript_model_ase.ckpt"
