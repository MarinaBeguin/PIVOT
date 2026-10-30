# PIVOT

[![CI](https://github.com/MarinaBeguin/PIVOT/actions/workflows/ci.yml/badge.svg)](https://github.com/MarinaBeguin/PIVOT/actions/workflows/ci.yml)

Film a room, then ask whether furniture fits and how to rearrange it. PIVOT rebuilds the room
in 3D, finds its objects with SAM 3.1, and a Muse (or Llama) agent answers by calling geometric
tools that verify every move: collisions, passages, ceiling height.

Built for the Meta Global AI Hackathon 2026. Full documentation to come.

## Checks

```bash
pip install -e ".[dev,serve]"
bash scripts/check.sh "$(dirname "$(command -v python)")"   # tests, mypy, pylint
```

## Docker

The image serves the viewer and the agent (API version: no GPU needed). The scene directory is
the output of `python -m pivot.build`; the Meta API key is given at run time, never stored in
the image.

```bash
docker build -t pivot .
docker run -p 8000:8000 -e META_API_KEY=... -v /path/to/scene:/scene:ro pivot
```
