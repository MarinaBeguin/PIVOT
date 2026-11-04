# Browser video uploads on Modal

`deploy/modal_app.py` registers the existing web demo and three new functions:

1. `video_capture`: MapAnything on one L40S, cameras and depth only.
2. `video_scene`: CPU/API privacy, inventory, segmentation and scene building.
3. `video_cleanup`: hourly cleanup of expired or deleted uploads.

The GPU is released before the API/CPU stage. No Slurm and no user's local PC
are involved after upload. The header contains a visible **Upload your video**
tab (translated with the page language). When uploads are disabled, the tab
remains visible and explains why file selection is unavailable.

## Scope and privacy

Uploads are enabled by default in the updated Modal deployment; set
`PIVOT_MODAL_UPLOADS=0` at deployment time to disable them. The existing public
URL does not change until this code is actually deployed.

Each uploaded job belongs to the browser's random HttpOnly upload cookie.
Status, scene pages, objects and tools all check that ownership. A private room
is not added to the shared public room menu. Its owner opens it through the
completion link and sees it alongside the public rooms. A restart can recover
job status and completed scene files; conversational edits remain in-memory.
Clearing cookies loses anonymous access to an uploaded room.

The original video and reconstructed frames are processed on Modal. Unblurred
capture frames are sent to Meta's privacy API; inventory/segmentation use the
blurred video afterward. Detection can miss sensitive details. No Meta key is
sent to the browser. Job ids, progress and safe failure messages are returned;
raw subprocess logs stay in deployment logs. The upload panel shows the
cloud/API processing notice and actual size/duration/retention settings.

Access expires after 24 hours by default. Hourly cleanup deletes expired data,
so automatic physical cleanup can occur up to an hour after expiry. The owner
can request deletion earlier: access is revoked, active calls are cancelled and
available payloads and web caches are removed. Stage-boundary checks and the
hourly cleaner remove leftovers if cancellation races with a worker.

## Independent limits and costs

These are **upload limits**, separate from PIVOT's conversation budget. They
bound accepted work but are not a dollar-denominated hard cap on Modal or Meta
billing. Configure provider spending controls for the account as appropriate.
Failed and deleted attempts still count toward admission quotas.

Set these shell environment variables when serving/deploying (a native Modal
CLI does not load deployment options from `.env`; that file supplies the Meta
secret). Docker `--env-file .env` passes its values to the deployment CLI.

| Variable | Default | Meaning |
|---|---|---|
| `PIVOT_MODAL_UPLOADS` | `1` | Enable remote uploads; `0` disables the backend |
| `PIVOT_UPLOAD_MAX_BYTES` | `100000000` | 100 MB, enforced while streaming |
| `PIVOT_UPLOAD_MAX_SECONDS` | `120` | Duration checked by ffprobe before GPU admission |
| `PIVOT_UPLOAD_FRAMES` | `150` | Frames reconstructed, allowed range 1–150 |
| `PIVOT_UPLOAD_OBJECT_KINDS` | `20` | Inventory cap, allowed range 1–20 |
| `PIVOT_UPLOAD_DAILY` | `4` | Global admissions per UTC day |
| `PIVOT_UPLOAD_PER_VISITOR` | `1` | Admissions per IP-derived visitor key per UTC day |
| `PIVOT_UPLOAD_ACTIVE` | `1` | Outstanding queued/running jobs admitted at once |
| `PIVOT_UPLOAD_RETENTION` | `86400` | Access expiry in seconds |
| `PIVOT_UPLOAD_NAMESPACE` | `pivot` | Prefix of persistent jobs/quota and model-cache volumes |

GPU and CPU functions each have a 15-minute runtime limit and one container;
application retries are disabled to avoid resubmitting paid inference silently.
Timed-out subprocess groups are killed. The web service remains at one
container: its lock plus persisted ledger serialize quota admission. Raising
web replica count requires a distributed admission mechanism first.

Media must contain a video stream, a finite positive duration, and dimensions
between 32 and 4096 pixels per side. Files are streamed to temporary disk, not
loaded entirely into the web container's memory. Exhausted quotas are checked
before reading a large body, then checked again when reserving a job.

## Prepare deployment

Use Modal SDK 1.6.3 (the declaration/import checks used this version) or build
the supplied deployment CLI image:

```bash
docker build -f docker/Dockerfile.optional --target deploy -t pivot-modal .
touch "$HOME/.modal.toml"
chmod 600 "$HOME/.modal.toml"
docker run --rm -it -v "$HOME/.modal.toml:/root/.modal.toml" pivot-modal token new
```

Keep `META_API_KEY` in the untracked `.env`. The deployment CLI forwards it to
Modal as a secret. The CLI image includes source, scripts, Docker constraints
and deployment helpers, without credentials or personal inputs.

Provide at least one already built scene (`scene.json` and `cloud.npz`) through
`PIVOT_DEMO_SCENES`. To keep all four public rooms when deploying, supply all
four existing exports; do not replace them with the smoke-test staircase.
Images and volumes are built/created only when the user runs Modal, not during
local CPU tests or SDK declaration checks.

## Test before publishing

From the worktree root, with a test scene at `data/room/scene`, launch an
isolated preview (this starts real paid Modal work if a video is submitted):

```bash
docker run --rm -it --env-file .env \
  -v "$HOME/.modal.toml:/root/.modal.toml" \
  -v "$PWD/data:/data:ro" \
  -e PIVOT_DEMO_SCENES="Example=/data/room/scene" \
  -e PIVOT_UPLOAD_NAMESPACE=pivot-preview \
  -e PIVOT_UPLOAD_PER_VISITOR=4 -e PIVOT_UPLOAD_DAILY=4 \
  pivot-modal serve deploy/modal_app.py
```

Use the preview URL printed by Modal, not the public production URL. Test a
single-level room video within the displayed bounds. Check the tab, processing
progress, opening the completed scene, geometry/chat, reload recovery, a second
browser's inability to open the private link, and deletion. Review the actual
GPU/CPU call states and deployment logs, not just a missing running job. Keep
the preview process open until the test finishes.

When approved, deploy with the same CLI invocation but `deploy` instead of
`serve`, production limits/namespace and all four public scene exports. That
updates the public app and can incur build, GPU, storage and API charges.

## Validation status

CPU tests cover quotas, deletion without quota refunds, cross-browser access
control, restart recovery, stream size enforcement, invalid-media rejection,
failed/missing outputs, and the worker chain/expiry cleanup with a fake SDK.
Chromium desktop/mobile tests exercise the upload tab and fake-job UI flow.
Real SDK declarations load four functions without deployment. None of those
checks proves real Modal image building, storage synchronization, GPU execution
or external API permissions. A real preview video remains the release gate.

The CPU image verifies OpenCV and the inventory module at build time. Upload jobs
require an actual inventory: failure stops processing rather than substituting
office objects. The default inventory cap matches the local API pipeline (20).

## Spoken replies

Modal deployments enable Meta MMS-TTS by default for English, French, German
and Spanish. The image downloads these voices at build time. Visitors enable
“Read answers aloud” in the interface; speech recognition remains Muse Voice
Transcribe. Use `PIVOT_TTS=browser` when serving/deploying to disable server TTS
and use the device voices. Unsupported languages also use browser voices.
