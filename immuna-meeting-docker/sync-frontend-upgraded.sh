#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="${DEST_DIR:-$ROOT_DIR/frontend}"
OVERLAY_DIR="${OVERLAY_DIR:-$ROOT_DIR/frontend-overlay-upgraded}"
DEFAULT_SOURCE="${IMMUNA_SOURCE:-/home/tobi/immuna}"
SOURCE_INPUT="${1:-$DEFAULT_SOURCE}"
UPSTREAM_REF="${IMMUNA_REF:-main}"
CACHE_DIR="${IMMUNA_CACHE_DIR:-$ROOT_DIR/.cache/upstream-immuna}"

if [ ! -d "$OVERLAY_DIR" ]; then
  echo "Overlay directory not found: $OVERLAY_DIR" >&2
  exit 1
fi

looks_like_git_source() {
  case "$1" in
    http://*|https://*|git@*|*.git) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_upstream_dir() {
  local source="$1"

  if [ -d "$source" ]; then
    printf '%s\n' "$source"
    return 0
  fi

  if ! looks_like_git_source "$source"; then
    echo "Upstream frontend source not found: $source" >&2
    echo "Pass a local checkout path or a git URL." >&2
    return 1
  fi

  mkdir -p "$(dirname "$CACHE_DIR")"

  if [ ! -d "$CACHE_DIR/.git" ]; then
    git clone --depth 1 --branch "$UPSTREAM_REF" "$source" "$CACHE_DIR"
  else
    git -C "$CACHE_DIR" fetch --depth 1 origin "$UPSTREAM_REF"
    git -C "$CACHE_DIR" checkout --force FETCH_HEAD
    git -C "$CACHE_DIR" clean -fd
  fi

  printf '%s\n' "$CACHE_DIR"
}

UPSTREAM_DIR="$(resolve_upstream_dir "$SOURCE_INPUT")"

mkdir -p "$DEST_DIR"

rsync -a --delete \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude 'dist' \
  --exclude '.DS_Store' \
  "$UPSTREAM_DIR"/ "$DEST_DIR"/

rsync -a "$OVERLAY_DIR"/ "$DEST_DIR"/

echo "Synced upstream frontend from: $UPSTREAM_DIR"
echo "Applied upgraded backend overlay from: $OVERLAY_DIR"
echo "Overlay files:"
find "$OVERLAY_DIR" -type f | sort | sed "s#^$OVERLAY_DIR/#  #"
