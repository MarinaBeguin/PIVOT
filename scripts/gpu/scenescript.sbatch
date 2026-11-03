#!/usr/bin/env bash
#SBATCH --job-name=scenescript
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
# Reads a room's walls, doors and windows with SceneScript, from its points.
#
#   sbatch scripts/gpu/scenescript.sbatch <points.csv.gz> <layout.txt>
#
# <points.csv.gz> comes from `python -m pivot.lifting.scenescript export`; <layout.txt>
# gets SceneScript's program, which `python -m pivot.lifting.scenescript import` reads
# back. Needs scripts/gpu/setup_scenescript_env.sh and the checkpoint (SCENESCRIPT_CKPT to
# point elsewhere).
set -euo pipefail
SCRATCH="${PIVOT_SCRATCH:-$HOME/pivot_scratch}"
DIR="$SCRATCH/scenescript"
cd "$DIR/code"
"$SCRATCH/conda_envs/scenescript/bin/python" - "$1" "$2" "${SCENESCRIPT_CKPT:-$DIR/scenescript_model_ase.ckpt}" <<'EOF'
import sys

from src.data.point_cloud import PointCloud
from src.networks.scenescript_model import SceneScriptWrapper

points, layout, checkpoint = sys.argv[1:4]
model = SceneScriptWrapper.load_from_checkpoint(checkpoint).cuda()
cloud = PointCloud.load_from_file(points)
print(f"{len(cloud.points)} points")
program = model.run_inference(cloud.points, nucleus_sampling_thresh=0.05, verbose=True)
text = program.generate_language_string()
open(layout, "w", encoding="utf-8").write(text + "\n")
print(text)
EOF
