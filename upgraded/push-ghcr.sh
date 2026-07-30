#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_API_IMAGE="${API_IMAGE:-photoguard-upgraded-api}"
LOCAL_WEB_IMAGE="${WEB_IMAGE:-photoguard-upgraded-web}"
GHCR_NAMESPACE="${GHCR_NAMESPACE:?Set GHCR_NAMESPACE to your GitHub user or org}"
GHCR_USERNAME="${GHCR_USERNAME:-$GHCR_NAMESPACE}"
GHCR_SCOPE="${GHCR_NAMESPACE,,}"
GHCR_API_NAME="${GHCR_API_NAME:-photoguard-upgraded-api}"
GHCR_WEB_NAME="${GHCR_WEB_NAME:-photoguard-upgraded-web}"
TAG="${TAG:-latest}"
PUSH_LATEST="${PUSH_LATEST:-0}"

REMOTE_API_IMAGE="ghcr.io/${GHCR_SCOPE}/${GHCR_API_NAME}:${TAG}"
REMOTE_WEB_IMAGE="ghcr.io/${GHCR_SCOPE}/${GHCR_WEB_NAME}:${TAG}"

cd "$ROOT_DIR"

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

if [ -n "${CR_PAT:-}" ]; then
  echo "$CR_PAT" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin
else
  echo "CR_PAT not set; assuming you are already logged in to ghcr.io"
fi

docker tag "$LOCAL_API_IMAGE" "$REMOTE_API_IMAGE"
docker tag "$LOCAL_WEB_IMAGE" "$REMOTE_WEB_IMAGE"
docker push "$REMOTE_API_IMAGE"
docker push "$REMOTE_WEB_IMAGE"

if [ "$PUSH_LATEST" = "1" ] && [ "$TAG" != "latest" ]; then
  docker tag "$LOCAL_API_IMAGE" "ghcr.io/${GHCR_SCOPE}/${GHCR_API_NAME}:latest"
  docker tag "$LOCAL_WEB_IMAGE" "ghcr.io/${GHCR_SCOPE}/${GHCR_WEB_NAME}:latest"
  docker push "ghcr.io/${GHCR_SCOPE}/${GHCR_API_NAME}:latest"
  docker push "ghcr.io/${GHCR_SCOPE}/${GHCR_WEB_NAME}:latest"
fi

echo "Pushed images:"
echo "  $REMOTE_API_IMAGE"
echo "  $REMOTE_WEB_IMAGE"

