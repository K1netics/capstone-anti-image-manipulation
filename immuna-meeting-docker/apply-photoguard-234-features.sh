#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="${1:-/home/tobi/immuna-meeting}"
TARGET_DIR="${2:-$ROOT_DIR/frontend}"
DOCKERIZED_FRONTEND_DIR="${DOCKERIZED_FRONTEND_DIR:-/home/tobi/photoguard/dockerized/frontend}"
OVERLAY_DIR="$ROOT_DIR/frontend-feature-overlay-234"

if [ ! -d "$SOURCE_REPO" ]; then
  echo "Source repo not found: $SOURCE_REPO" >&2
  exit 1
fi

if [ ! -d "$DOCKERIZED_FRONTEND_DIR/src" ]; then
  echo "Dockerized frontend source not found: $DOCKERIZED_FRONTEND_DIR" >&2
  exit 1
fi

"$ROOT_DIR/sync-frontend.sh" "$SOURCE_REPO"

mkdir -p "$TARGET_DIR/src/components/home" "$TARGET_DIR/src/lib" "$TARGET_DIR/src/types" "$TARGET_DIR/public"

# Copy the proven 2.3.4-compatible functional files directly from the dockerized frontend.
cp "$DOCKERIZED_FRONTEND_DIR/src/components/home/MaskEditor.tsx" "$TARGET_DIR/src/components/home/MaskEditor.tsx"
cp "$DOCKERIZED_FRONTEND_DIR/src/components/home/OutputSection.tsx" "$TARGET_DIR/src/components/home/OutputSection.tsx"
cp "$DOCKERIZED_FRONTEND_DIR/src/lib/image.ts" "$TARGET_DIR/src/lib/image.ts"
cp "$DOCKERIZED_FRONTEND_DIR/src/types/api.ts" "$TARGET_DIR/src/types/api.ts"
cp "$DOCKERIZED_FRONTEND_DIR/src/vite-env.d.ts" "$TARGET_DIR/src/vite-env.d.ts"

# Apply local overlay files for runtime API_UPSTREAM support and minimal-visual-change page wiring.
rsync -a "$OVERLAY_DIR"/ "$TARGET_DIR"/

echo "Applied PhotoGuard 2.3.4 feature overlay to: $TARGET_DIR"
echo "Feature source: $DOCKERIZED_FRONTEND_DIR"
