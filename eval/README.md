# Evaluation

Does PIVOT answer requests about a real room correctly, and does it ever claim a
move works when it does not? Every request below has a known answer, computed on
the room's geometry; every answer is replayed by a judge with exact geometry.

## Sets

| Set | Room | Requests | Used for |
|---|---|---|---|
| [dev/office](dev/office) | The office of the demo | 36, 6 kinds | Developing PIVOT: its 3 errors on a first run were studied and fixed, so its score is optimistic |
| heldout/bedroom | Bedroom | About 40, 10 kinds | Testing only |
| heldout/classroom | Classroom | About 40, 10 kinds | Testing only |
| heldout/apartment | Apartment (several rooms) | About 40, 10 kinds | Testing only |

The held-out requests were generated once (seed 1; a request drawn twice is
replaced or dropped) and are run once, after the last change to PIVOT's agent
and tools: no code changes after seeing how they were answered. The three rooms
are in the demo, and the 3D building was tuned on them; the requests, and how
the agent answers them, were never seen during development.

The requests name objects by number, so they only hold for the scene they were
made on: `scene.sha256` records it, and the script refuses another one.

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

Development set (office, 36 requests):

| Variant | Success (95% CI) | False claims | False refusals | Latency p50 |
|---|---|---|---|---|
| Muse alone | 53% (37–68%) | 15/36 | 2/36 | 20 s |
| Muse + a final check | 86% (71–94%) | 3/36 | 2/36 | 41 s |
| **PIVOT** | **100% (90–100%)** | **0/36** | **0/36** | 14 s |
| Geometric search alone | 97% (86–100%) | 1/36 | 0/36 | 2 s |

Held-out set: made and run once on the rooms of the demo as they are finally
built, every variant at the same time, with Muse and with open weights.

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
