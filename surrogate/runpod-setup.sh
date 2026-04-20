#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PHOTOGUARD_VENV:-/workspace/venvs/pg-surrogate-cu128}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TMP_REQS="/tmp/pg-surrogate-api-reqs-no-torch.txt"

mkdir -p "$(dirname "${VENV_DIR}")"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip wheel setuptools
python -m pip uninstall -y torch torchvision torchaudio xformers >/dev/null 2>&1 || true

grep -Ev '^(torch|torchvision|xformers)($|[<>=!~])' "${ROOT_DIR}/api/requirements.txt" > "${TMP_REQS}"

python -m pip install --index-url "${PYTORCH_INDEX_URL}" torch==2.8.0 torchvision==0.23.0
python -m pip install -r "${TMP_REQS}"
python -m pip install --no-deps facenet-pytorch

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no-gpu")
PY
