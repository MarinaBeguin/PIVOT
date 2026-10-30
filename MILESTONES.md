# PIVOT milestones

Meta Global AI Developer Hackathon, indie track. Build window Oct 27 – Nov 4, 2026
(submission due Nov 4, 9 PM ET = Nov 5, 03:00 Zurich).

## Pitch

Film a room for 30 seconds. PIVOT rebuilds it in 3D, finds every object, and lets you ask
"Can I move this chair next to the window?". A Muse agent answers by calling geometric tools
(collision checks, placement search, motion planning) that act as a verifier: the language
model never makes up geometry, and every answer it gives is backed by a check.

## Pipeline

```text
video ──► keyframes ──► 3D reconstruction ──► gravity-aligned metric frame
              │                                         │
              ├──► Muse Spark: object vocabulary        │
              └──► SAM 3.1: tracked masks per noun ─────┴──► lift masks to 3D
                                                              │
                    scene graph (objects, oriented boxes, carved voxel volumes, floor map)
                                                              │
                  Muse Wonder agent ◄── tool calls ──► geometry engine (verifier)
                                                              │
                                                     interactive 3D viewer
```

## Milestones

| # | Milestone | Done when | Target | Status |
|---|---|---|---|---|
| M0 | Repo, tooling, data access | `scripts/check.sh` passes; dev scenes reachable | Oct 27 | done |
| M1 | Geometry core | Boxes, collisions, floor map, placement checks and search unit-tested | Oct 27 | done |
| M1b | Motion verification | `plan_motion` finds collision-free (x, y, yaw) paths, re-validated exactly; distinguishes `NO_PATH` from `SEARCH_LIMIT_REACHED` | Oct 27 | done |
| M2 | Scene from ground truth | Lifting validated on synthetic views; then on real masks of a ScanNet++ scene: boxes reprojected on images, floor offset measured against the mesh | Oct 28 | real scene built (self-hosted SAM 3.1); mesh comparison pending |
| M3 | Meta perception | Muse Spark proposes the vocabulary; SAM 3.1 returns tracked masks; results cached on disk | Oct 28 | SAM 3.1 via API done (9 prompts, ~36 min sequential); vocabulary (`pivot.perception.inventory`: Muse Spark, or Llama 3.2 Vision for the open version) to test on the cluster |
| M2b | Real volumes | Objects carved from depth (free space seen through), thin legs kept via masks, shapes disjoint; collisions and motion planning slab by slab | Oct 28 | done |
| M5 | Verified agent | Goal → action → geometric check → feedback → revised action loop; tools: placement, placement search, motion planning; model id configurable (Spark vs Wonder) | Oct 29–30 | |
| M6 | Interactive viewer | Browser viewer: scene, chat panel, animated verified motions | Oct 31–Nov 1 | local viewer + server done (pick, check, move, undo, animation); chat panel and Modal deployment pending |
| M7 | Evaluation | Synthetic task suite; variants: LLM alone, geometric planner alone, LLM + final check, full PIVOT; same geometric-check budget for the last two | Nov 2 | `pivot.eval` done; 30 tasks on the real scene (20 feasible); planner run local, model variants to run on the cluster |
| M3b | Two versions | API version: SAM 3.1 and Muse through the Meta API (`pivot.meta.sam`, `pivot.agent`). Open-weight version, no key: SAM 3.1 self-hosted (`sam3_local`) and Llama served by vLLM (`scripts/cluster/llama_server.sbatch`) | Oct 30–31 | code done; one script per version (`scripts/run_api.sh`, `scripts/run_open.sh`): object list, SAM 3.1 in parallel with a progress bar, detections cached per video and backend; Llama to test on the cluster |
| M4 | Reconstruction from a personal video | Off-the-shelf Meta model (VGGT or MapAnything) gives poses and depth for a phone video without LiDAR; no new reconstruction method | Nov 2–3 | optional |
| M8 | Submission | Public repo, README, 3-minute demo video, X post | Nov 3–4 | |

Feature freeze: Nov 2. If time runs short, motion verification and the agent loop come before visual polish.

### Known limitations (documented, not hidden)

- Objects slide and rotate about the vertical axis only; they are never tilted or lifted.
- Objects resting on a moved object follow it rigidly; their stability is not simulated.
- `NO_PATH` means no path exists at the lattice resolution (5 cm, 22.5 degrees), not a proof of impossibility.
- Walls are inferred as non-floor areas of the floor map; unsegmented clutter is not yet an obstacle.
- Volumes are 4 cm cells: thin parts (chair tubes, some legs) can be missing, and space never seen
  through (behind a monitor) is kept as occupied.
- Phone poses (ARKit) are kept as is: against the laser scan, a quarter of the frames are off by
  more than 5 cm, in bursts; self-supervised refinement (frame and segment ICP, pose graph) did not
  help on real data and was removed.
- Tracks are merged into one object only if SAM never separated them in a shared frame (disjoint
  masks of at least 0.5% of the image); boxes of neighboring objects may still overlap, their
  volumes do not.
- Parts hidden under furniture are guessed, not observed: an object smaller than the minimum size of
  its label (chairs, stools, trash cans) is extended under what hides it, continuing its visible
  part at the same heights; a much smaller fragment touching an object of the same label is taken
  for a misplaced far view of it and dropped.
- Objects not named in the SAM prompts are not detected.
- Doors and windows are snapped to the nearest side of the floor rectangle: right for rectangular
  rooms, approximate otherwise. A track whose masks lie inside another label's masks is taken as a
  part of that object (a lamp head segmented as a "shelf").
- Only furniture labeled as a surface (table, desk, shelf, cabinet...) receives objects put down on it.
- Voxel aliasing: a single intermediate pose overlapping by at most one cell (5 cm), between two
  free poses, is accepted along a motion; a real contact lasts longer.
- Making room can still take about 1.5 min in hard cases (large object, several replans).

## Judging alignment

| Criterion | Weight | How PIVOT addresses it |
|---|---|---|
| Impact | 30% | Room planning and accessibility checks on a real home from a phone video |
| Meta AI tooling | 25% | SAM 3.1 video tracking, Muse Spark vision for the vocabulary, Muse Wonder agent with tool calling |
| Demo | 25% | Live: film, reconstruct, ask, see the verified move |
| Technical depth | 20% | Multi-view mask lifting, placement and motion verification, ablation on a task suite, tested and typed codebase |

## Engineering rules

- `pytest`, `mypy --strict` and `pylint` stay green; Google-style docstrings.
- An experiment that doesn't help is deleted, not kept behind a flag.
- Secrets only in `.env`.
