# Use CUDA 13 runtime with cuDNN on Ubuntu 22.04
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ca-certificates curl build-essential \
    libsndfile1 ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# make python3 available as python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# copy app + artifacts
# - local_inpaint_model: saved diffusers pipeline
# - backend: app code (app.py, utils.py, etc.)
# optional artifacts: torch_cu130.whl, requirements-freeze.txt
COPY ./artifacts/local_inpaint_model /app/local_inpaint_model
COPY ./backend /app/backend
COPY ./artifacts/requirements-freeze.txt /app/requirements-freeze.txt

# Upgrade pip and install torch wheel if available; otherwise install from nightly cu130 index
RUN python -m pip install --upgrade pip wheel setuptools && \
    if [ -f /app/torch_cu130.whl ]; then \
        python -m pip install --pre --upgrade "torch" "torchvision" --index-url https://download.pytorch.org/whl/nightly/cu130; \
    else \
        python -m pip install --pre --upgrade "torch" "torchvision" --index-url https://download.pytorch.org/whl/nightly/cu130; \
    fi

# Install Python requirements
# If you provided requirements-freeze.txt, prefer installing from it; otherwise install common deps
RUN if [ -f /app/requirements-freeze.txt ]; then \
        python -m pip install -r /app/requirements-freeze.txt || true; \
    else \
        python -m pip install diffusers transformers accelerate safetensors ftfy gradio pillow numpy requests xformers || true; \
    fi


# Upgrade pip
RUN python -m pip install --upgrade pip wheel setuptools

# Install torch (cu130 nightly)
RUN python -m pip install --pre --upgrade torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu130

# Install required libraries
RUN python -m pip install \
    diffusers \
    transformers \
    accelerate \
    safetensors \
    ftfy \
    gradio \
    pillow \
    numpy \
    requests \
    xformers
# Expose Gradio port
EXPOSE 7860

# set environment variables for huggingface cache to persistent location inside container
ENV HF_HOME=/root/.cache/huggingface
ENV XDG_CACHE_HOME=/root/.cache
ENV PHOTOGUARD_INPAINT_MODEL=/app/local_inpaint_model
ENV PHOTOGUARD_LOCAL_FILES_ONLY=1

# run from the backend directory so relative assets like ./images keep working
WORKDIR /app/backend
CMD ["python", "app.py"]
