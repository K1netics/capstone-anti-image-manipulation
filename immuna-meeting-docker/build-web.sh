#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
WEB_IMAGE="${WEB_IMAGE:-immuna:1.0}"
OCI_SOURCE="${OCI_SOURCE:-$(git -C "$PROJECT_ROOT" config --get remote.backup.url || git -C "$PROJECT_ROOT" config --get remote.origin.url || true)}"

docker build \
  --build-arg OCI_SOURCE="$OCI_SOURCE" \
  -t "$WEB_IMAGE" \
  -f "$ROOT_DIR/web.Dockerfile" \
  "$PROJECT_ROOT"

echo "Built image: $WEB_IMAGE"
