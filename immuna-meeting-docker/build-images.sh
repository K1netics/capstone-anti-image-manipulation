#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
API_IMAGE="${API_IMAGE:-photoguard-api}"
WEB_IMAGE="${WEB_IMAGE:-photoguard-web}"
OCI_SOURCE="${OCI_SOURCE:-$(git -C "$PROJECT_ROOT" config --get remote.origin.url || true)}"

docker build \
  --build-arg OCI_SOURCE="$OCI_SOURCE" \
  -t "$API_IMAGE" \
  -f "$PROJECT_ROOT/upgraded/docker/backend.Dockerfile" \
  "$PROJECT_ROOT/upgraded"

docker build \
  --build-arg OCI_SOURCE="$OCI_SOURCE" \
  -t "$WEB_IMAGE" \
  -f "$ROOT_DIR/web.Dockerfile" \
  "$PROJECT_ROOT"

echo "Built images:"
echo "  API: $API_IMAGE"
echo "  WEB: $WEB_IMAGE"
