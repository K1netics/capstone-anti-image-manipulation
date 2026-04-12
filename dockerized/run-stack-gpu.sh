#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_IMAGE="${API_IMAGE:-photoguard-api}"
WEB_IMAGE="${WEB_IMAGE:-photoguard-web}"
API_CONTAINER="${API_CONTAINER:-photoguard-api}"
WEB_CONTAINER="${WEB_CONTAINER:-photoguard-web}"
NETWORK_NAME="${NETWORK_NAME:-photoguard-net}"
WEB_PORT="${WEB_PORT:-8080}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/../artifacts/local_inpaint_model}"
LOCAL_FILES_ONLY="${PHOTOGUARD_LOCAL_FILES_ONLY:-1}"

if [ ! -d "$MODEL_DIR" ]; then
  echo "Model directory not found: $MODEL_DIR" >&2
  echo "Set MODEL_DIR or place the model at ../artifacts/local_inpaint_model" >&2
  exit 1
fi

if ! docker image inspect "$API_IMAGE" >/dev/null 2>&1; then
  echo "Missing image: $API_IMAGE" >&2
  echo "Run ./build-images.sh first." >&2
  exit 1
fi

if ! docker image inspect "$WEB_IMAGE" >/dev/null 2>&1; then
  echo "Missing image: $WEB_IMAGE" >&2
  echo "Run ./build-images.sh first." >&2
  exit 1
fi

docker network inspect "$NETWORK_NAME" >/dev/null 2>&1 || docker network create "$NETWORK_NAME" >/dev/null
docker rm -f "$API_CONTAINER" "$WEB_CONTAINER" >/dev/null 2>&1 || true

docker run -d \
  --name "$API_CONTAINER" \
  --network "$NETWORK_NAME" \
  --network-alias api \
  --gpus all \
  -e PHOTOGUARD_INPAINT_MODEL=/app/models/local_inpaint_model \
  -e PHOTOGUARD_LOCAL_FILES_ONLY="$LOCAL_FILES_ONLY" \
  -v "$MODEL_DIR:/app/models/local_inpaint_model:ro" \
  "$API_IMAGE" >/dev/null

docker run -d \
  --name "$WEB_CONTAINER" \
  --network "$NETWORK_NAME" \
  -e API_UPSTREAM="${API_UPSTREAM:-http://api:8000}" \
  -p "$WEB_PORT:80" \
  "$WEB_IMAGE" >/dev/null

echo "PhotoGuard is running at http://localhost:$WEB_PORT"
echo "GPU mode enabled for API container: $API_CONTAINER"
