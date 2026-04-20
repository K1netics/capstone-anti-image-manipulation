#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PHOTOGUARD_VENV:-/workspace/venvs/pg-surrogate-cu128}"

if [[ -x "${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"
fi

MODEL_DIR="${PHOTOGUARD_MODEL_DIR:-/workspace/artifacts/sdxl_local_inpaint_model}"
if [[ ! -f "${MODEL_DIR}/model_index.json" ]]; then
  for candidate in \
    /workspace/models/sdxl_local_inpaint_model \
    /workspace/artifacts/sdxl_local_inpaint_model \
    /workspace/local_inpaint_model \
    /workspace/models/local_inpaint_model
  do
    if [[ -f "${candidate}/model_index.json" ]]; then
      MODEL_DIR="${candidate}"
      break
    fi
  done
fi

export PHOTOGUARD_MODEL_DIR="${MODEL_DIR}"
export PHOTOGUARD_LOCAL_FILES_ONLY="${PHOTOGUARD_LOCAL_FILES_ONLY:-1}"
export PHOTOGUARD_ENABLE_XFORMERS="${PHOTOGUARD_ENABLE_XFORMERS:-1}"
export PHOTOGUARD_ENABLE_UNET_CHECKPOINTING="${PHOTOGUARD_ENABLE_UNET_CHECKPOINTING:-0}"
export PHOTOGUARD_ENABLE_VAE_CHECKPOINTING="${PHOTOGUARD_ENABLE_VAE_CHECKPOINTING:-0}"
export PHOTOGUARD_REQUIRE_FULL_STRENGTH="${PHOTOGUARD_REQUIRE_FULL_STRENGTH:-1}"
export PHOTOGUARD_DEFAULT_IMMUNIZATION_PROFILE="${PHOTOGUARD_DEFAULT_IMMUNIZATION_PROFILE:-surrogate_deevid}"
export PHOTOGUARD_COMPILE="${PHOTOGUARD_COMPILE:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"

echo "Starting surrogate backend"
echo "  model_dir=${PHOTOGUARD_MODEL_DIR}"
echo "  local_files_only=${PHOTOGUARD_LOCAL_FILES_ONLY}"
echo "  xformers=${PHOTOGUARD_ENABLE_XFORMERS}"
echo "  unet_checkpointing=${PHOTOGUARD_ENABLE_UNET_CHECKPOINTING}"
echo "  vae_checkpointing=${PHOTOGUARD_ENABLE_VAE_CHECKPOINTING}"
echo "  default_profile=${PHOTOGUARD_DEFAULT_IMMUNIZATION_PROFILE}"
echo "  python=$(command -v python)"

exec python -m uvicorn api.main:app --app-dir "${ROOT_DIR}" --host 0.0.0.0 --port 8000
