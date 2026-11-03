#!/usr/bin/env bash
# Serves the open-weight agent's model with an OpenAI-compatible API (vLLM):
# Llama 3.1 8B by default, or Meta's Muse Glimmer 30B (bf16, about 60 GB: two
# 48 GB GPUs in tensor parallel), with its own tool-call parser.
#
#   sbatch scripts/gpu/llama_server.sbatch [model]
#   sbatch --gres=gpu:2 --mem=96G scripts/gpu/llama_server.sbatch meta-models/Muse-Glimmer-30B
#
# The log prints the address to use, e.g. http://<node>:8000/v1; then, on the machine
# running PIVOT:
#   export META_API_BASE_URL=http://<node>:8000/v1 META_API_KEY=local \
#          PIVOT_AGENT_MODEL=<model> PIVOT_REASONING_EFFORT=
#   python -m pivot.agent --scene ... "..."      (or pivot.server --chat, pivot.eval)
#SBATCH --job-name=pivot-llama
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
case "$MODEL" in
  *Muse-Glimmer*) PARSER=muse_glimmer ;;
  *) PARSER=llama3_json ;;
esac
# One process per GPU the job got.
GPUS=$(nvidia-smi --list-gpus | wc -l)
ENV="${PIVOT_LLAMA_ENV:-${PIVOT_SCRATCH:-$HOME/pivot_scratch}/conda_envs/pivot_llama}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${PIVOT_SCRATCH:-$HOME/pivot_scratch}/hf_cache}" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "OpenAI-compatible API at http://$(hostname):8000/v1 serving $MODEL"
# The agent's conversations reach ~25k tokens (object list and tool results).
exec "$ENV/bin/vllm" serve "$MODEL" --host 0.0.0.0 --port 8000 --max-model-len 32768 \
  --tensor-parallel-size "$GPUS" --enable-auto-tool-choice --tool-call-parser "$PARSER"
