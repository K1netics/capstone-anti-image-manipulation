FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ARG OCI_SOURCE=""
ARG PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    git \
    ca-certificates \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY api/requirements.txt /app/api/requirements.txt

RUN grep -Ev '^(torch|torchvision|xformers)($|[<>=!~])' /app/api/requirements.txt > /tmp/requirements-no-torch.txt && \
    python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch torchvision && \
    python -m pip install -r /tmp/requirements-no-torch.txt && \
    python -m pip install xformers || true

COPY api /app/api
COPY backend /app/backend

RUN mkdir -p /app/models/local_inpaint_model

ENV HF_HOME=/root/.cache/huggingface \
    XDG_CACHE_HOME=/root/.cache \
    PHOTOGUARD_MODEL_DIR=/app/models/local_inpaint_model \
    PHOTOGUARD_LOCAL_FILES_ONLY=0 \
    PHOTOGUARD_REQUIRE_FULL_STRENGTH=1 \
    PHOTOGUARD_ENABLE_XFORMERS=1 \
    PHOTOGUARD_COMPILE=0

LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.description="PhotoGuard upgraded FastAPI backend for SDXL-capable GPU deployments"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

