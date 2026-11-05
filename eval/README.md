# Evaluation

Does PIVOT answer requests about a real room correctly, and does it ever claim a
move works when it does not? Every request below has a known answer, computed on
the room's geometry; every answer is replayed by a judge with exact geometry.

## Recorded sets

| Set | Room | Requests | Status |
|---|---|---|---|
| [dev/office](dev/office) | Earlier office | 36, 6 kinds | Development set; errors were studied and fixed |
| [Regenerated office](results/2026-11-04-regenerated/office) | Office | 39 | Included in the latest four-scene run |
| [Regenerated bedroom](results/2026-11-04-regenerated/bedroom) | Bedroom | 25 | Included in the latest four-scene run |
| [Regenerated classroom](results/2026-11-04-regenerated/classroom) | Classroom | 40 | Included in the latest four-scene run |
| [Regenerated apartment](results/2026-11-04-regenerated/apartment) | Apartment | 35 | Included in the latest four-scene run |

The latest archive contains 139 tasks and 556 unique task/variant records. Request ids
refer to particular objects, so tasks must use their matching scene. Scene fingerprints
were checked against the run's archived `scene.json` files before copying results here.
Do not substitute a freshly reconstructed scene, even if its room name is the same.

The office was used during development and the other rooms were used for reconstruction
tuning. The run has no recorded launch code commit or exact model version; do not describe
it as a verified frozen-code held-out experiment. The `heldout` names used by the launcher
express the intended task split, not proof of an independently unseen-room protocol.

## Kinds of requests

| Kind | Example | Right when |
|---|---|---|
| `next_to` | Move the chair (#56) next to the table (#21). | Within 30 cm of it, reached by a collision-free path |
| `next_to_clear` | ... Keep doors and windows clear. | The same, and nothing in front of a door or window |
| `at_point` | Move the desk (#1) so that its center is at x = 1.5 m, y = 2.5 m. | Within 15 cm of the point; half the points are on other furniture, so often impossible |
| `tuck_under` | Push the chair (#3) under the desk (#1). | At least 30% of it under the top |
| `on_surface` | Put the lamp on the desk. | Resting on its top |
| `at_point_blocked` | Move the shelf to (x, y), other objects in the way. | There, the others moved aside and put back |
| `to_room` | Move the armchair (#12) into the bedroom. | In that room, through the doors |
| `turn_in_place` | Turn the shelf (#4) a quarter turn where it stands. | Turned by 90° (± 15°), its center within 15 cm of where it was |
| `clear_door` | The chair stands in front of a door: move it. | No longer in front of any door |
| `against_wall` | Put the desk (#1) against a wall. | Within 10 cm of a wall |
| `wheelchair_way` | Make room for a wheelchair at the table (#24). | A 90 cm passage from a door to a free spot at the table |

A request with no solution must be refused; a refusal of a feasible one is a
*false refusal*, a claim that an impossible move works a *false claim*.

## Variants

| Variant | What answers |
|---|---|
| `llm_only` | Muse Spark alone: it sees the objects and gives a final position, unchecked |
| `llm_check` | Muse Spark proposes, a motion planner says only "feasible" or not, Muse tries again |
| `pivot` | The full agent: Muse Spark with PIVOT's geometric tools |
| `planner_only` | No language model: the geometric search, given the request already parsed |

`planner_only` is a ceiling, not a rival: it gets the request in structured
form, and the labels come from the same planner, so its score mostly checks
that the judge and the planner agree. It cannot read a request in words.

## Metrics

Success rate with its 95% Wilson interval; false claims; false refusals;
expected cost (a false claim counts 10 times a false refusal); depth of the
collisions of `llm_only`'s answers; latency (median and 95th percentile);
model calls and tokens per request. `summary.json` holds them all, by kind too.

## Results

See the [main results tables](../README.md#results) and the
[archived run](results/2026-11-04-regenerated) for all original records and the audit:

- PIVOT: **134/139 (96.4%)**, three erroneous success claims, two false refusals.
- Muse alone: **47/139 (33.8%)** with explicit errors counted as failures; the unmodified
  original summaries report 48/139 because one timeout is counted as a correct refusal.
- Muse + final check: **115/139 (82.7%)**.
- Geometric search alone: **139/139**, with structured tasks and planner-derived labels.
- Separate recorded safety contracts: **41 passed, 1 expected failure**; synthetic scenes
  and scripted dialogue, not a live-model or browser evaluation.

Open-weight launchers are provided below, but this archive contains only the API variants.
No open-weight performance is inferred from these numbers. The earlier 36-task office
result is historical and development-biased; it is not substituted for the latest run.

Recompute the archived audit without API calls:

```bash
python eval/results/2026-11-04-regenerated/audit.py
```

The audit validates complete task/variant coverage, task kinds/labels and original summary
counts. Its adjusted aggregate counts explicit `error:` outcomes as failures without
modifying the original records. It writes `audit.json` and file checksums. Scene/cloud
snapshots are not included in the Git archive; `scene.sha256` identifies each required scene.

The demo itself is checked by hand before a release or a recording, on a laptop
and on a phone: [demo-checklist.md](demo-checklist.md).

## Running it

```bash
bash eval/run_heldout.sh                 # every variant on the three rooms
bash eval/run_heldout.sh planner_only    # geometry only, no API key
PIVOT_EVAL_OPEN=llama bash eval/run_heldout.sh     # open weights, no API key: Llama 3.1 8B
PIVOT_EVAL_OPEN=glimmer bash eval/run_heldout.sh   # or Muse Glimmer 30B (two GPUs)
```

With open weights, the model runs as a GPU job (Slurm) and its variants are
recorded as `llm_only_llama`, `pivot_llama` (or `_glimmer`), next to the Muse ones:
the same verifier, without the API.

The model variants need `META_API_KEY` in `.env`; the four variants on the three
rooms cost about $15 to $20 of API calls. Scenes are read from
`$PIVOT_RUNS/<room>/scene`. An interrupted run resumes where it stopped. Each
room gets `results.jsonl` (one line per request and variant), `summary.md` and
`summary.json`.

One room by hand:

```bash
python -m pivot.eval generate --scene <scene_dir> --tasks tasks.json --count 44 --seed 1
python -m pivot.eval run --scene <scene_dir> --tasks tasks.json --out results.jsonl
python -m pivot.eval summary --out results.jsonl --json summary.json
```
