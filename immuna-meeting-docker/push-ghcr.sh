#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_API_IMAGE="${API_IMAGE:-photoguard-api}"
LOCAL_WEB_IMAGE="${WEB_IMAGE:-photoguard-web}"
GHCR_REGISTRY="${GHCR_REGISTRY:-ghcr.io}"
GHCR_NAMESPACE="${GHCR_NAMESPACE:?Set GHCR_NAMESPACE to your GitHub user or org}"
GHCR_USERNAME="${GHCR_USERNAME:-$GHCR_NAMESPACE}"
GHCR_SCOPE="${GHCR_NAMESPACE,,}"
GHCR_API_NAME="${GHCR_API_NAME:-photoguard-api}"
GHCR_WEB_NAME="${GHCR_WEB_NAME:-photoguard-web}"
TAG="${TAG:?Set TAG to a version, e.g. TAG=v0.1.0}"
PUSH_LATEST="${PUSH_LATEST:-0}"
GHCR_TOKEN_VALUE="${CR_PAT:-${GHCR_TOKEN:-}}"

REMOTE_API_IMAGE="${GHCR_REGISTRY}/${GHCR_SCOPE}/${GHCR_API_NAME}:${TAG}"
REMOTE_WEB_IMAGE="${GHCR_REGISTRY}/${GHCR_SCOPE}/${GHCR_WEB_NAME}:${TAG}"
REMOTE_API_LATEST="${GHCR_REGISTRY}/${GHCR_SCOPE}/${GHCR_API_NAME}:latest"
REMOTE_WEB_LATEST="${GHCR_REGISTRY}/${GHCR_SCOPE}/${GHCR_WEB_NAME}:latest"

cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required but was not found in PATH." >&2
  exit 1
fi

if ! docker image inspect "$LOCAL_API_IMAGE" >/dev/null 2>&1; then
  echo "Missing local image: $LOCAL_API_IMAGE" >&2
  echo "Run ./build-images.sh first." >&2
  exit 1
fi

if ! docker image inspect "$LOCAL_WEB_IMAGE" >/dev/null 2>&1; then
  echo "Missing local image: $LOCAL_WEB_IMAGE" >&2
  echo "Run ./build-images.sh first." >&2
  exit 1
fi

if [ -n "$GHCR_TOKEN_VALUE" ]; then
  printf '%s' "$GHCR_TOKEN_VALUE" | docker login "$GHCR_REGISTRY" -u "$GHCR_USERNAME" --password-stdin
else
  echo "CR_PAT/GHCR_TOKEN not set; assuming you are already logged in to $GHCR_REGISTRY"
fi

docker tag "$LOCAL_API_IMAGE" "$REMOTE_API_IMAGE"
docker tag "$LOCAL_WEB_IMAGE" "$REMOTE_WEB_IMAGE"
docker push "$REMOTE_API_IMAGE"
docker push "$REMOTE_WEB_IMAGE"

if [ "$PUSH_LATEST" = "1" ] && [ "$TAG" != "latest" ]; then
  docker tag "$LOCAL_API_IMAGE" "$REMOTE_API_LATEST"
  docker tag "$LOCAL_WEB_IMAGE" "$REMOTE_WEB_LATEST"
  docker push "$REMOTE_API_LATEST"
  docker push "$REMOTE_WEB_LATEST"
fi

echo "Pushed images:"
echo "  $REMOTE_API_IMAGE"
echo "  $REMOTE_WEB_IMAGE"
if [ "$PUSH_LATEST" = "1" ] && [ "$TAG" != "latest" ]; then
  echo "  $REMOTE_API_LATEST"
  echo "  $REMOTE_WEB_LATEST"
fi
