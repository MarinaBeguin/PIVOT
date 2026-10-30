# Shared by run_api.sh and run_open.sh: arguments, object list, segmentation
# (Meta API or our GPUs) with cache and progress bar, scene building and
# serving. Sourced, not run.

usage() {
  cat >&2 <<EOF
usage: $0 [--quality fast|medium|full] <scannetpp_scene_dir> <video.mp4> <out_dir> [prompt ...]
  Builds the scene of a ScanNet++ capture and serves it at http://127.0.0.1:\$PORT.
  --quality  share of the video's frames SAM 3.1 tracks: fast (1/3), medium (1/2) or
             full (all, the default). Faster, but objects seen briefly may be missed.
  Without prompts, a vision model lists the objects the video shows.
EOF
  exit 2
}

QUALITY=full
while [ "$#" -gt 0 ] && [[ "$1" == --* ]]; do
  case "$1" in
    --quality) [ "$#" -ge 2 ] || usage; QUALITY="$2"; shift 2 ;;
    *) usage ;;
  esac
done
case "$QUALITY" in
  fast) STEP=3 ;;
  medium) STEP=2 ;;
  full) STEP=1 ;;
  *) usage ;;
esac
[ "$#" -ge 3 ] || usage
CAPTURE="$1"; VIDEO="$2"; OUT="$3"; shift 3
# Objects SAM 3.1 looks for: given on the command line, or listed by a vision
# model looking at the video (see choose_prompts).
PROMPTS=("$@")
# Read by the GPU jobs, which run a copy of their script.
export PIVOT_SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SLURM_CONF="${SLURM_CONF:-/home/sladmcvl/slurm/slurm.conf}"
# Where the open weights are (speech model of the server, models of the GPU jobs).
export HF_HUB_CACHE="${HF_HUB_CACHE:-/itet-stor/mbeguin/net_scratch/hf_cache}"
PYTHON="${PIVOT_PYTHON:-/itet-stor/mbeguin/net_scratch/conda_envs/pivot_sam3/bin/python}"
PORT="${PORT:-8765}"
LOGS=/itet-stor/mbeguin/net_scratch/pivot/logs
CACHE="${PIVOT_CACHE:-/itet-stor/mbeguin/net_scratch/pivot/cache}"
# The video keeps one frame out of VIDEO_STRIDE of the 60 fps capture.
VIDEO_STRIDE="${PIVOT_VIDEO_STRIDE:-10}"
SCENE="$OUT/scene"
[ "$STEP" -eq 1 ] || SCENE="$OUT/scene_$QUALITY"
mkdir -p "$OUT"

# Slurm jobs and background requests, stopped on exit (Ctrl-C included).
JOBS=()
cleanup() {
  [ "${#JOBS[@]}" -eq 0 ] || scancel "${JOBS[@]}"
  kill $(jobs -p) 2>/dev/null || true
}
trap cleanup EXIT

video_id() {
  # Names a video by its file name, size and date, so that caches follow it.
  echo "$(basename "${VIDEO%.*}")_$(stat -L -c '%s %Y' "$VIDEO" | md5sum | cut -c1-8)"
}

detections_dir() {
  # Detections depend on the video, the frames tracked and the SAM backend only,
  # so they are cached per video: any run on the same video reuses them.
  local suffix=""
  [ "$STEP" -eq 1 ] || suffix="_every$STEP"
  echo "$CACHE/$1/$(video_id)$suffix"
}

sam_video() {
  # The video SAM 3.1 tracks: the input, or one frame out of STEP of it.
  local video="$CACHE/videos/$(video_id)_every$STEP.mp4"
  if [ "$STEP" -eq 1 ]; then
    echo "$VIDEO"
    return
  fi
  [ -f "$video" ] || (cd "$PIVOT_SRC" && "$PYTHON" -m pivot.data.video subsample "$VIDEO" \
    "$STEP" "$video")
  echo "$video"
}

has_api_key() {
  # True if a Meta API key is set, in the environment or in the .env file.
  [ -n "${META_API_KEY:-}" ] || grep -qE '^META_API_KEY=.+' "$PIVOT_SRC/.env" 2>/dev/null
}

file_stem() {
  # The file name pivot gives a prompt's detections, without extension.
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/_/g; s/^_|_$//g'
}

prompts_done_in() {
  # True if a detection directory holds every prompt: those of the command
  # line, else those of the object list stored there.
  local dir="$1" prompts=("${PROMPTS[@]}") prompt
  if [ "${#prompts[@]}" -eq 0 ]; then
    [ -s "$dir/prompts.txt" ] || return 1
    mapfile -t prompts < "$dir/prompts.txt"
  fi
  for prompt in "${prompts[@]}"; do
    [ -f "$dir/$(file_stem "$prompt").npz" ] || return 1
  done
}

choose_prompts() {
  # Fills PROMPTS unless given on the command line: from the list cached with the
  # detections, else by running "$@" (the backend's inventory) to write it.
  local list="$DETECTIONS/prompts.txt"
  mkdir -p "$DETECTIONS"
  if [ "${#PROMPTS[@]}" -eq 0 ]; then
    [ -s "$list" ] || "$@" || true
    if [ -s "$list" ]; then
      mapfile -t PROMPTS < "$list"
    else
      echo "   the vision model gave no list; using the default office objects" >&2
      PROMPTS=(chair table monitor lamp "trash can" door window shelf cabinet)
    fi
  fi
  local joined
  printf -v joined '%s, ' "${PROMPTS[@]}"
  echo "   ${#PROMPTS[@]} kinds of objects: ${joined%, }"
  [ ! -f "$DETECTIONS/prompts.jpg" ] || echo "   frames the model saw: $DETECTIONS/prompts.jpg"
}

list_objects_api() {
  # Muse Spark lists the objects (seconds).
  (cd "$PIVOT_SRC" && PIVOT_BACKEND=api "$PYTHON" -m pivot.perception.inventory --backend api \
    --video "$VIDEO" --out "$DETECTIONS/prompts.txt" --env-file "$PIVOT_SRC/.env" \
    --max-objects "${PIVOT_MAX_OBJECTS_API:-20}")
}

list_objects_local() {
  # Llama 3.2 Vision lists the objects on a GPU (minutes, mostly loading the weights).
  JOBS=("$(sbatch --parsable "$PIVOT_SRC/scripts/cluster/inventory.sbatch" "$VIDEO" \
    "$DETECTIONS/prompts.txt" "${PIVOT_MAX_OBJECTS_OPEN:-8}")")
  echo "   log: $LOGS/inventory_${JOBS[0]}.out"
  while squeue -h -j "${JOBS[0]}" 2>/dev/null | grep -q .; do sleep 5; done
  JOBS=()
}

missing_prompts() {
  # Prints, one per line, the prompts without a detection file in $DETECTIONS.
  local prompt
  for prompt in "${PROMPTS[@]}"; do
    [ -f "$DETECTIONS/$(file_stem "$prompt").npz" ] || echo "$prompt"
  done
}

worker_prompts() {
  # Fills WORKER_PROMPTS with the missing prompts of worker $1 out of $2 (round robin).
  local index="$1" workers="$2" i=0 prompt
  WORKER_PROMPTS=()
  while IFS= read -r prompt; do
    [ $((i % workers)) -eq "$index" ] && WORKER_PROMPTS+=("$prompt")
    i=$((i + 1))
  done < <(missing_prompts)
}

announce() {
  # Says how many objects are left to track and how long it should take, for
  # each quality. Measured on 704 frames: 0.46 s per frame and prompt on an
  # A6000 plus 40 s of loading, 0.34 s through the API.
  local backend="$1" workers="$2" video="$3" missing frames
  missing=$(missing_prompts | wc -l)
  if [ "$missing" -eq 0 ]; then
    echo "   all ${#PROMPTS[@]} kinds already tracked (cache)"
    return
  fi
  frames=$(cd "$PIVOT_SRC" && "$PYTHON" -m pivot.data.video frames "$video" 2>/dev/null) \
    || frames=""
  if [ -z "$frames" ]; then
    echo "   $missing kinds of objects to track, $workers at a time"
    return
  fi
  awk -v n="$missing" -v w="$workers" -v f="$((frames * STEP))" -v q="$QUALITY" \
      -v api="$([ "$backend" = api ] && echo 1 || echo 0)" 'BEGIN {
    rate = api ? 0.34 : 0.46; load = api ? 0 : 40
    rounds = int((n + w - 1) / w)
    split("full medium fast", names); split("1 2 3", steps)
    for (i = 1; i <= 3; i++) minutes[names[i]] = (load + rounds * f / steps[i] * rate) / 60
    printf "   %d kinds of objects to track, %d at a time: about %.0f min", n, w, minutes[q]
    printf "%s\n", api ? "" : " once the GPUs are free"
    printf "   (--quality full: %.0f min, medium: %.0f min, fast: %.0f min)\n", \
      minutes["full"], minutes["medium"], minutes["fast"]
  }'
}

show_progress() {
  # One line, redrawn in place: the share of prompts done (the trackers write a
  # .progress file in permille while a prompt runs) and the prompts in progress.
  local prompt stem permille=0 current=() width=40 percent filled bar
  for prompt in "${PROMPTS[@]}"; do
    stem="$DETECTIONS/$(file_stem "$prompt")"
    if [ -f "$stem.npz" ]; then
      permille=$((permille + 1000))
    elif [ -f "$stem.progress" ]; then
      permille=$((permille + $(cat "$stem.progress" 2>/dev/null || echo 0)))
      current+=("$prompt")
    fi
  done
  percent=$((permille / (10 * ${#PROMPTS[@]})))
  filled=$((percent * width / 100))
  bar=$(printf "%${filled}s" "" | tr ' ' '|')
  local doing="$1"
  if [ "${#current[@]}" -gt 0 ]; then
    printf -v doing '%s, ' "${current[@]}"
    doing="${doing%, }"
  fi
  printf "\r   Recognizing objects [%-${width}s] %3d%% (%s)\033[K" "$bar" "$percent" "$doing"
}

finish_progress() {
  # Ends the progress line; fails if a prompt has no detections.
  show_progress ""
  echo
  if [ -n "$(missing_prompts)" ]; then
    echo "Segmentation failed for: $(missing_prompts | paste -sd, -); see $1" >&2
    exit 1
  fi
}

segment_api() {
  # SAM 3.1 through the Meta API, several prompts at a time.
  local workers="${PIVOT_SAM_WORKERS:-5}" video worker
  video=$(sam_video)
  announce api "$workers" "$video"
  for ((worker = 0; worker < workers; worker++)); do
    worker_prompts "$worker" "$workers"
    [ "${#WORKER_PROMPTS[@]}" -eq 0 ] || (cd "$PIVOT_SRC" && PIVOT_BACKEND=api exec "$PYTHON" \
      -m pivot.meta.sam --video "$video" --out "$DETECTIONS" --prompts "${WORKER_PROMPTS[@]}") \
      >"$DETECTIONS/worker_$worker.log" 2>&1 &
  done
  # The API returns a prompt's tracks at once: the bar moves prompt by prompt.
  while [ -n "$(jobs -rp)" ]; do
    show_progress "tracking"
    sleep 5
  done
  finish_progress "$DETECTIONS/worker_*.log"
}

segment_local() {
  # Self-hosted SAM 3.1, one GPU job per group of prompts.
  local workers="${PIVOT_SAM_WORKERS:-3}" video worker
  video=$(sam_video)
  announce local "$workers" "$video"
  for ((worker = 0; worker < workers; worker++)); do
    worker_prompts "$worker" "$workers"
    [ "${#WORKER_PROMPTS[@]}" -eq 0 ] || JOBS+=("$(sbatch --parsable \
      "$PIVOT_SRC/scripts/cluster/sam3_video.sbatch" "$video" "$DETECTIONS" "${WORKER_PROMPTS[@]}")")
  done
  if [ "${#JOBS[@]}" -gt 0 ]; then
    while squeue -h -j "$(IFS=,; echo "${JOBS[*]}")" 2>/dev/null | grep -q .; do
      show_progress "waiting for a GPU"
      sleep 5
    done
  fi
  JOBS=()
  finish_progress "$LOGS/sam3_<job id>.out"
}

build_scene() {
  # Lifting, volumes and floor map run on the CPU (a few minutes).
  if [ ! -f "$SCENE/scene.json" ]; then
    (cd "$PIVOT_SRC" && "$PYTHON" -m pivot.build --scene "$CAPTURE" \
      --detections "$DETECTIONS" --out "$SCENE" --stride $((VIDEO_STRIDE * STEP)) \
      --prompts "${PROMPTS[@]}")
  else
    echo "   already built: $SCENE"
  fi
}

serve() {
  echo "== Serving http://127.0.0.1:$PORT (from your laptop: ssh -L $PORT:127.0.0.1:$PORT <host>)"
  cd "$PIVOT_SRC"
  "$PYTHON" -m pivot.server --scene-dir "$SCENE" --port "$PORT" --chat
}
