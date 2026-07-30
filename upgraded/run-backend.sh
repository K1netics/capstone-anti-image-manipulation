#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-1}"

if [ "${PHOTOGUARD_MODEL_DIR:-}" = "" ] && [ -d "$ROOT_DIR/artifacts/sdxl_local_inpaint_model" ]; then
  export PHOTOGUARD_MODEL_DIR="$ROOT_DIR/artifacts/sdxl_local_inpaint_model"
fi

if [ "${PHOTOGUARD_LOCAL_FILES_ONLY:-}" = "" ]; then
  export PHOTOGUARD_LOCAL_FILES_ONLY=1
fi

if [ "${PHOTOGUARD_ENABLE_XFORMERS:-}" = "" ]; then
  export PHOTOGUARD_ENABLE_XFORMERS=1
fi

if [ "${PHOTOGUARD_REQUIRE_FULL_STRENGTH:-}" = "" ]; then
  export PHOTOGUARD_REQUIRE_FULL_STRENGTH=1
fi

ARGS=(
  api.main:app
  --app-dir "$ROOT_DIR"
  --host "$HOST"
  --port "$PORT"
)

if [ "$RELOAD" = "1" ]; then
  ARGS+=(--reload)
fi

exec python -m uvicorn "${ARGS[@]}"
