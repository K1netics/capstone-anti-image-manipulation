#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_TAR="${INPUT_TAR:-$ROOT_DIR/photoguard-images.tar}"

if [ ! -f "$INPUT_TAR" ]; then
  echo "Bundle not found: $INPUT_TAR" >&2
  exit 1
fi

docker load -i "$INPUT_TAR"

echo "Loaded images from:"
echo "  $INPUT_TAR"
