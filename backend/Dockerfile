FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04

ARG OCI_SOURCE=""

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
    libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

COPY backend/requirements.txt /app/backend/requirements.txt

RUN python -m pip install --upgrade pip wheel setuptools && \
    python -m pip install --pre --upgrade torch torchvision \
      --index-url https://download.pytorch.org/whl/nightly/cu130 && \
    python -m pip install -r /app/backend/requirements.txt && \
    python -m pip install xformers || true

COPY backend/app /app/backend/app

RUN mkdir -p /app/models

ENV HF_HOME=/root/.cache/huggingface \
    XDG_CACHE_HOME=/root/.cache \
    PHOTOGUARD_INPAINT_MODEL=/app/models/local_inpaint_model \
    PHOTOGUARD_LOCAL_FILES_ONLY=1

LABEL org.opencontainers.image.source="${OCI_SOURCE}" \
      org.opencontainers.image.description="PhotoGuard FastAPI image generation backend"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "/app/backend", "--host", "0.0.0.0", "--port", "8000"]
