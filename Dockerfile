# Server-only (default): docker build -t pivot .
# GPU tools: docker build --target tools -t pivot-tools .
FROM python:3.12-slim AS server
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[serve,fast]"
COPY scripts ./scripts
COPY docker ./docker
RUN useradd --create-home --uid 1000 pivot && mkdir /data /cache && chown pivot:pivot /data /cache
ENV PIVOT_BACKEND=api PIVOT_SCENE_DIR=/scene PORT=8000 \
    PIVOT_EXECUTOR=local PIVOT_PYTHON=/usr/local/bin/python \
    PIVOT_SCRATCH=/cache HF_HOME=/cache/huggingface TORCH_HOME=/cache/torch \
    PIVOT_UPLOAD_DIR=/data/uploads PIVOT_HOST=0.0.0.0
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ['PORT']+'/api/objects', timeout=4)"
USER pivot
CMD ["bash", "docker/serve.sh"]

# One CUDA stack for MapAnything, SAM, vision inventory, speech and guards.
FROM server AS tools
USER root
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY docker/constraints.txt /opt/pivot-constraints.txt
ENV PIP_CONSTRAINT=/opt/pivot-constraints.txt
RUN pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
COPY docker/share_torch.py /opt/share_torch.py
COPY docker/map-constraints.txt /opt/map-constraints.txt
RUN python -m venv /opt/map-env && python /opt/share_torch.py
ARG MAPANYTHING_REV=3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9
ARG SAM3_REV=f1647e8d0c1d280991f4d1b7d974c2704537fe66
RUN git clone https://github.com/facebookresearch/map-anything.git /opt/map-anything \
    && git -C /opt/map-anything checkout "$MAPANYTHING_REV" \
    && PIP_CONSTRAINT=/opt/map-constraints.txt /opt/map-env/bin/python -m pip install /opt/map-anything /app \
    && /opt/map-env/bin/python -m pip check
RUN git clone https://github.com/facebookresearch/sam3.git /opt/sam3 \
    && git -C /opt/sam3 checkout "$SAM3_REV" \
    && pip install /opt/sam3
RUN pip install 'transformers>=4.45,<5' einops psutil pycocotools scikit-image scikit-learn matplotlib \
    imageio-ffmpeg opencv-python-headless==4.10.0.84 'scipy<1.17' '.[guard]' && pip check
# Check imports at build time; weights and GPU execution are checked at runtime.
RUN python -c "import torch, torchvision, transformers; from sam3.model_builder import build_sam3_multiplex_video_predictor; assert torch.version.cuda == '12.6'" \
    && /opt/map-env/bin/python -c "import torch; from mapanything.models import MapAnything; assert torch.version.cuda == '12.6'"
ENV PIVOT_CAPTURE_PYTHON=/opt/map-env/bin/python
USER pivot

# Keep the default image small and CPU-only.
FROM server AS default
