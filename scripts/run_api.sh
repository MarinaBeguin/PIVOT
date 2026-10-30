#!/usr/bin/env bash
# PIVOT, API version: SAM 3.1 and Muse through the Meta Model API. No GPU needed;
# the API key is read from .env (META_API_KEY).
#
#   bash scripts/run_api.sh [--quality fast|medium|full] <scannetpp_scene_dir> <video.mp4> \
#       <out_dir> [prompt ...]
#
# Without prompts, Muse Spark lists the objects the video shows (up to
# PIVOT_MAX_OBJECTS_API, 20). Segments the video (PIVOT_SAM_WORKERS requests in
# parallel, 5), builds the scene, then serves the viewer with the agent. Steps
# already done are skipped: object list and detections are cached per video,
# the scene per out_dir.
set -euo pipefail
source "$(dirname "$0")/pipeline_common.sh"
export PIVOT_BACKEND=api
DETECTIONS=$(detections_dir sam_api)

echo "== [1/3] Listing the objects of the video with Muse Spark"
choose_prompts list_objects_api
echo "== [2/3] Recognizing objects with SAM 3.1 through the Meta API ($QUALITY quality)"
segment_api
echo "== [3/3] Building the 3D scene"
build_scene
serve
