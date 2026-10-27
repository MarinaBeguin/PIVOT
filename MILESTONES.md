# PIVOT milestones

Meta Global AI Developer Hackathon, indie track. Build window Oct 27 – Nov 4, 2026
(submission due Nov 4, 9 PM ET = Nov 5, 03:00 Zurich).

## Pitch

Film a room for 30 seconds. PIVOT rebuilds it in 3D, finds every object, and lets you ask
"Can I move this chair next to the window?". A Muse agent answers by calling geometric tools
(collision checks, free-space search, clearance paths) that act as a verifier: the language
model never makes up geometry, and every answer it gives is backed by a check.

## Pipeline

```text
video ──► keyframes ──► 3D reconstruction ──► gravity-aligned metric frame
              │                                         │
              ├──► Muse Spark: object vocabulary        │
              └──► SAM 3.1: tracked masks per noun ─────┴──► lift masks to 3D
                                                              │
                                     scene graph (objects, oriented boxes, floor map)
                                                              │
                  Muse Wonder agent ◄── tool calls ──► geometry engine (verifier)
                                                              │
                                                     interactive 3D viewer
```

## Milestones

| # | Milestone | Done when | Target |
|---|---|---|---|
| M0 | Repo, tooling, data access | `pytest`, `mypy`, `pylint` pass; dev scenes reachable | Oct 27 |
| M1 | Geometry core | Oriented boxes, collisions, floor map, placement search are unit-tested on synthetic scenes | Oct 27–28 |
| M2 | Scene from ground truth | A ScanNet++ scene loaded with GT poses and depth, lifted to a scene graph from masks | Oct 28 |
| M3 | Meta perception | Muse Spark proposes the vocabulary; SAM 3.1 returns tracked masks; results are cached on disk | Oct 28–29 |
| M4 | Reconstruction from raw video | Poses and depth estimated from video on the cluster GPU replace GT; metric scale checked against ScanNet++ | Oct 29–30 |
| M5 | Verified agent | A Muse Wonder tool-calling loop answers placement questions; every claim is backed by a tool result | Oct 30–31 |
| M6 | Interactive viewer | Browser viewer shows the scene, chat panel, proposed moves rendered live | Nov 1–2 |
| M7 | Personal video end-to-end and polish | User video goes through the full pipeline; stretch goals below | Nov 2–3 |
| M8 | Submission | Public repo, README, 3-minute demo video, X post | Nov 3–4 |

### Stretch goals (only if M0–M6 are solid)

- Gaussian splatting of the scene with per-object Gaussians that move when the agent moves an object.
- Voice commands through Muse Voice Transcribe.
- Accessibility audit: "Is there a 90 cm wheelchair path from the door to the bed?"

## Judging alignment

| Criterion | Weight | How PIVOT addresses it |
|---|---|---|
| Impact | 30% | Room planning and accessibility checks on a real home from a phone video |
| Meta AI tooling | 25% | SAM 3.1 video tracking, Muse Spark vision for the vocabulary, Muse Wonder agent with tool calling |
| Demo | 25% | Live: film, reconstruct, ask, see the verified move |
| Technical depth | 20% | Multi-view mask lifting, geometric verifier, tested and typed codebase |

## Engineering rules

- `pytest`, `mypy --strict` and `pylint` stay green; Google-style docstrings.
- An experiment that doesn't help is deleted, not kept behind a flag.
- Secrets only in `.env`.
