"""Run upstream SceneScript in its own environment."""
import argparse
from pathlib import Path

from src.data.point_cloud import PointCloud
from src.networks.scenescript_model import SceneScriptWrapper

parser = argparse.ArgumentParser()
parser.add_argument('points')
parser.add_argument('layout', type=Path)
parser.add_argument('--checkpoint', default='/weights/scenescript_model_ase.ckpt')
args = parser.parse_args()
if not Path(args.checkpoint).is_file():
    parser.error('Mount the SceneScript checkpoint at --checkpoint')
model = SceneScriptWrapper.load_from_checkpoint(args.checkpoint).cuda()
cloud = PointCloud.load_from_file(args.points)
program = model.run_inference(cloud.points, nucleus_sampling_thresh=0.05, verbose=True)
args.layout.write_text(program.generate_language_string() + '\n', encoding='utf-8')
