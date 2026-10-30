# PIVOT web app: the 3D viewer and the agent, served by FastAPI (API version:
# Muse and SAM 3.1 through the Meta API, geometry on the CPU, no GPU needed).
#
#   docker build -t pivot .
#   docker run -p 8000:8000 -e META_API_KEY=... -v /path/to/scene:/scene:ro pivot
#
# The scene directory is the output of `python -m pivot.build` (scene.json and
# cloud.npz). The API key is given at run time and never stored in the image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
# Dependencies first, so that code changes do not reinstall them.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve]"

RUN useradd --create-home --uid 1000 pivot
USER pivot

ENV PIVOT_BACKEND=api \
    PIVOT_SCENE_DIR=/scene \
    PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/api/objects', timeout=4)"
# PORT can be set by the host (Railway, Cloud Run...).
CMD ["sh", "-c", "exec python -m pivot.server --scene-dir \"$PIVOT_SCENE_DIR\" --host 0.0.0.0 --port \"$PORT\" --chat"]
