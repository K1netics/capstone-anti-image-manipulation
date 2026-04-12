#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_IMAGE="${API_IMAGE:-photoguard-api}"
WEB_IMAGE="${WEB_IMAGE:-photoguard-web}"
OCI_SOURCE="${OCI_SOURCE:-}"

cd "$ROOT_DIR"

docker build \
  --build-arg OCI_SOURCE="$OCI_SOURCE" \
  -t "$API_IMAGE" \
  -f backend/Dockerfile \
  .
docker build \
  --build-arg OCI_SOURCE="$OCI_SOURCE" \
  -t "$WEB_IMAGE" \
  -f proxy/Dockerfile \
  .

echo "Built images:"
echo "  API: $API_IMAGE"
echo "  WEB: $WEB_IMAGE"
