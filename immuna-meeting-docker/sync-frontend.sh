#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="${1:-/home/tobi/immuna-meeting}"
DEST_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/frontend"
OVERLAY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/frontend-overlay"

mkdir -p "$DEST_DIR"

rsync -a --delete \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude 'dist' \
  --exclude '.DS_Store' \
  "$SRC_DIR"/ "$DEST_DIR"/

if [ -d "$OVERLAY_DIR" ]; then
  rsync -a "$OVERLAY_DIR"/ "$DEST_DIR"/
fi

echo "Synced frontend from $SRC_DIR to $DEST_DIR"
