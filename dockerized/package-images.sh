#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_IMAGE="${API_IMAGE:-photoguard-api}"
WEB_IMAGE="${WEB_IMAGE:-photoguard-web}"
OUTPUT_TAR="${OUTPUT_TAR:-$ROOT_DIR/photoguard-images.tar}"

"$ROOT_DIR/build-images.sh"

docker save -o "$OUTPUT_TAR" "$API_IMAGE" "$WEB_IMAGE"

echo "Saved image bundle to:"
echo "  $OUTPUT_TAR"
