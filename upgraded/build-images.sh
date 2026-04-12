#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_IMAGE="${API_IMAGE:-photoguard-upgraded-api}"
WEB_IMAGE="${WEB_IMAGE:-photoguard-upgraded-web}"
OCI_SOURCE="${OCI_SOURCE:-$(git -C "$ROOT_DIR" config --get remote.origin.url || true)}"

cd "$ROOT_DIR"

docker build \
  --build-arg OCI_SOURCE="$OCI_SOURCE" \
  -t "$API_IMAGE" \
  -f docker/backend.Dockerfile \
  .
docker build \
  --build-arg OCI_SOURCE="$OCI_SOURCE" \
  -t "$WEB_IMAGE" \
  -f docker/proxy.Dockerfile \
  .

echo "Built images:"
echo "  API: $API_IMAGE"
echo "  WEB: $WEB_IMAGE"

