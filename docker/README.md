# Installation and tools

Linux with Docker Engine and Docker Compose v2. GPU workloads additionally need
the NVIDIA driver and NVIDIA Container Toolkit on the host. Conda and a host
CUDA toolkit are not required. Follow NVIDIA's installation instructions:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

## Prepare secrets and persistent storage

From the repository root:

```bash
cp .env.example .env
chmod 600 .env
mkdir -p data weights
```

Edit `.env` locally: set `META_API_KEY` for Muse, API SAM, photo processing and
API speech. Do not commit it. Add `HF_TOKEN` only for gated model downloads,
after accepting each model's terms. Model weights are downloaded at runtime,
not baked into images. Cache downloads in the named `pivot-cache` volume.
Mount data under `/data`, checkpoints under `/weights`; these are independent
of the host's username, home directory or cluster paths. Set `PIVOT_DATA_DIR`
if data live elsewhere. Bind-mounted outputs must be writable by container
UID 1000; if needed use `docker compose run --user "$(id -u):$(id -g)" ...`
and provide a writable cache mount too. Do not remove the cache volume to stop
containers: `down -v` deletes downloaded weights.

## Images and coverage

| Image / target | Tools | Credentials / weights |
|---|---|---|
| `server` (default) | Viewer, geometric planning, agent, evaluation, API segmentation, photo, speech | Meta key for API features; an existing scene for serving |
| `tools` | Above plus MapAnything, local SAM 3.1, Llama Vision inventory, wav2vec speech, local guards; CoTracker loads through torch.hub when requested | MapAnything; gated SAM / Llama weights; internet on first download |
| `docker/Dockerfile.llm` | vLLM endpoint for the local agent | Gated Llama weights; compatible NVIDIA driver for this independent image |
| `docker/Dockerfile.scenescript` | Upstream SceneScript inference | Separately obtained checkpoint; noncommercial upstream terms |
| `docker/Dockerfile.optional`, target `speech` | Omnilingual ASR package | Its checkpoints; PIVOT adapter remains in `night/open-weights` |
| Same, target `privacy` | EgoBlur CLI | Face/plate checkpoints; PIVOT adapter remains in `night/open-weights` |
| Same, target `deploy` | Modal deployment CLI | Modal authentication; only needed for deployment |

Muse Spark, Muse Image and Muse Voice are API services: no local model weights
are installed for these. Optional photo cleanup remains in `night/photo-cleanup`;
building an image does not merge that feature. Muse Glimmer uses `docker/Dockerfile.glimmer`, a separate vLLM 0.32.0
image with its tool and reasoning parsers. Its driver requirements are separate;
it is not covered by the CUDA 12.6 tools image. Its BF16 30B weights exceed one
48 GB GPU; the recipe defaults to two GPUs. Weights and inference remain untested. SceneScript and the optional packages have independent
environments; they do not replace the tools image's PyTorch.

## GPU preflight

```bash
docker compose --profile tools build tools
docker compose --profile tools run --rm tools bash docker/tools.sh check
```

This checks imports and an actual CUDA tensor calculation. The tools image pins
PyTorch 2.8 / CUDA 12.6 and constrains transitive dependencies to keep that stack.
MapAnything has its own Python environment with NumPy 2; SAM uses NumPy 1.26.
Only pinned PyTorch/CUDA package files are shared between these environments.
`PIVOT_CAPTURE_PYTHON` selects the reconstruction interpreter; `PIVOT_PYTHON`
selects the SAM/server interpreter.
A passing image build alone does not validate CUDA or model inference. Start
with 8 frames before a full video, then increase to 30 and 150 after inspection.
48 GB is not a guarantee that 150 frames fit for every input.
To pre-download selected checkpoints into the persistent cache:

```bash
docker compose --profile tools run --rm tools bash docker/tools.sh fetch-models \
  facebook/map-anything-apache
```

For local SAM use `facebook/sam3.1`; for local vision inventory use
`meta-llama/Llama-3.2-11B-Vision-Instruct`; for wav2vec speech use
`facebook/wav2vec2-large-960h-lv60-self`. Gated downloads require `HF_TOKEN` and
accepted model terms. CoTracker may also fetch source and weights through
`torch.hub` on first use; an HF cache alone is not an offline installation.

## Phone video, end to end on one PC

```bash
cp ~/PIVOT-video-iphone.MOV data/phone.MOV
PIVOT_CAPTURE_FRAMES=8 docker compose --profile tools run --rm --service-ports tools \
  bash docker/tools.sh video /data/phone.MOV /data/phone-test
```

Open http://localhost:8000 after reconstruction, privacy processing, object
segmentation and scene building. These steps call Meta and consume API credit.
No Slurm is used. Outputs are in `data/phone-test`; use a NEW output directory
when changing the input or frame count, because completed stages are reused.
Privacy blurs detections, not every possible sensitive detail; the privacy
SAM service receives the original video to find those detections.

For a real browser upload test, first serve an existing scene:

```bash
PIVOT_UPLOADS=1 docker compose --profile tools run --rm --service-ports \
  -e PIVOT_UPLOADS=1 -e PIVOT_SCENE_DIR=/data/phone-test/room/scene \
  tools bash docker/tools.sh serve
```

Choose "Send your video". The same local pipeline runs in the background and
adds the room after completion. Keep this container running until it finishes.
Jobs are tracked in memory; restarting the container does not resume browser
job tracking. Uploaded files and outputs persist in `data/uploads`.

## Existing captures and individual tools

```bash
# ScanNet++ capture plus its video -> API pipeline, then viewer.
docker compose --profile tools run --rm --service-ports tools \
  bash scripts/run_api.sh /data/capture /data/video.mp4 /data/room
# Local GPU perception (no Slurm):
docker compose --profile tools run --rm tools bash docker/tools.sh inventory \
  --video /data/video.mp4 --out /data/prompts.txt
docker compose --profile tools run --rm tools bash docker/tools.sh segment \
  --video /data/video.mp4 --out /data/detections --prompts chair table
# Geometry, evaluation, layout import/export, and other CLIs:
docker compose --profile tools run --rm tools python -m pivot.build --help
docker compose run --rm server python -m pivot.eval --help
docker compose run --rm server python -m pivot.lifting.scenescript --help
```

`run_open.sh` is still the cluster orchestrator. On one PC, invoke the local
inventory/segment/build stages above and run the LLM separately; do not launch
the cluster script inside Docker. On a single 48 GB GPU, stop perception before
starting the LLM to release its GPU memory.

## Local language model

```bash
docker compose --profile llm build llm
```

The Llama image uses vLLM 0.8.5.post1 in its own environment (see
https://docs.vllm.ai/en/v0.8.5.post1/getting_started/installation/gpu.html).
The vLLM image has its own entrypoint. Its actual CUDA preflight is:

```bash
docker compose --profile llm run --rm --entrypoint python llm \
  -c 'import torch; print(torch.ones(1, device="cuda"))'
docker compose --profile llm up llm
```

Wait for the model to finish loading. PIVOT containers use
`PIVOT_BACKEND=open`, `PIVOT_OPEN_URL=http://llm:8000/v1` and
`PIVOT_AGENT_MODEL=meta-llama/Llama-3.1-8B-Instruct`. On the host, use
http://localhost:8001/v1. The tools image CUDA version does not determine the
vLLM image's driver requirements: run the separate preflight before downloading
weights. This path is independent of Meta API services.

## SceneScript

The upstream environment compiles torchsparse and may take substantial time.
Set `SCENESCRIPT_REV` at build time for a particular source revision.

```bash
docker compose --profile scenescript build scenescript
docker compose --profile scenescript run --rm scenescript \
  /data/points.csv.gz /data/layout.txt \
  --checkpoint /weights/scenescript_model_ase.ckpt
```

Export points and import the resulting program with `pivot.lifting.scenescript`.
Checkpoints are not downloaded automatically.

## Other model tools and deployment

```bash
docker build -f docker/Dockerfile.optional --target speech -t pivot-speech .
docker build -f docker/Dockerfile.optional --target privacy -t pivot-privacy .
docker run --rm -v "$(pwd)/data:/data" -v "$(pwd)/weights:/weights:ro" \
  pivot-privacy --help
docker build -f docker/Dockerfile.optional --target deploy -t pivot-modal .
```

The optional speech and privacy images deliberately use CPU PyTorch builds:
Omnilingual ASR 0.2.0 uses Python 3.11, torch 2.8 and the official fairseq2 CPU
variant; PyArrow/pandas are pinned for NumPy 1.26 compatibility.
EgoBlur 2.0.1 requires torch 2.9.1.
They therefore do not silently install a newer CUDA stack on this host.
Optional images prepare the packages, not the unmerged PIVOT adapters. Test those
adapters in their own branches before claiming end-to-end support. Modal is not
needed for local use. Authenticate Modal separately and mount its configuration
at runtime; never bake API keys or authentication files into an image.

## Validation status

Verified during installation review: server image build and HTTP smoke test;
tools image build, real SAM/MapAnything imports, both environments' `pip check`;
EgoBlur image build and CLI help; Omnilingual ASR image build and inference-pipeline
import. No weights were downloaded for these checks. Llama, Glimmer and
SceneScript recipes have not been built or exercised in this review.

The local runner is covered by a CPU regression checking paths with spaces,
frame-count forwarding, privacy, downstream stride and cached-stage reuse.
Image builds, GPU preflight, checkpoint loading and a complete phone-video run
are distinct checks. Until they pass on the target machine, the recipes are
installation candidates, not validated end-to-end results. Optional image
versions should be locked further after the first successful model tests.

## Muse Glimmer (separate machine/runtime)

```bash
docker build -f docker/Dockerfile.glimmer -t pivot-glimmer .
```

This recipe checks the model's tool-parser import at build time. It is a
preparation recipe, not a validated model run. Before weights are available,
check its CUDA runtime with `--entrypoint python`, as for the Llama image.
Use a suitable driver and sufficient GPUs, then provide `HF_TOKEN`, a persistent
cache and the model checkpoint. Parser/model documentation:
https://docs.vllm.ai/en/v0.32.0/api/vllm/model_executor/models/muse_glimmer/
https://docs.vllm.ai/en/v0.32.0/api/vllm/tool_parsers/muse_glimmer_tool_parser/
