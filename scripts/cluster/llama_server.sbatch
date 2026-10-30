#!/usr/bin/env bash
# Serves Llama with an OpenAI-compatible API (vLLM), for the open-weight agent.
#
#   export SLURM_CONF=/home/sladmcvl/slurm/slurm.conf
#   sbatch scripts/cluster/llama_server.sbatch [model]
#
# The log prints the address to use, e.g. http://octopus04:8000/v1; then on hades:
#   export META_API_BASE_URL=http://<node>:8000/v1 META_API_KEY=local \
#          PIVOT_AGENT_MODEL=<model> PIVOT_REASONING_EFFORT=
#   python -m pivot.agent --scene ... "..."      (or pivot.server --chat, pivot.eval)
#SBATCH --job-name=pivot-llama
#SBATCH --partition=gpu.all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/itet-stor/mbeguin/net_scratch/pivot/logs/llama_%j.out
set -euo pipefail

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
ENV="${PIVOT_LLAMA_ENV:-/itet-stor/mbeguin/net_scratch/conda_envs/pivot_llama}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/itet-stor/mbeguin/net_scratch/hf_cache}" HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "OpenAI-compatible API at http://$(hostname):8000/v1 serving $MODEL"
# The agent's conversations reach ~25k tokens (object list and tool results).
exec "$ENV/bin/vllm" serve "$MODEL" --host 0.0.0.0 --port 8000 --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser llama3_json
