#!/usr/bin/env bash
set -euo pipefail

GHCR_NAMESPACE="${GHCR_NAMESPACE:?Set GHCR_NAMESPACE to your GitHub user or org}"
GHCR_SCOPE="${GHCR_NAMESPACE,,}"
GHCR_API_NAME="${GHCR_API_NAME:-photoguard-api}"
GHCR_WEB_NAME="${GHCR_WEB_NAME:-photoguard-web}"
LOCAL_API_IMAGE="${API_IMAGE:-photoguard-api}"
LOCAL_WEB_IMAGE="${WEB_IMAGE:-photoguard-web}"
TAG="${TAG:-latest}"

REMOTE_API_IMAGE="ghcr.io/${GHCR_SCOPE}/${GHCR_API_NAME}:${TAG}"
REMOTE_WEB_IMAGE="ghcr.io/${GHCR_SCOPE}/${GHCR_WEB_NAME}:${TAG}"

docker pull "$REMOTE_API_IMAGE"
docker pull "$REMOTE_WEB_IMAGE"

docker tag "$REMOTE_API_IMAGE" "$LOCAL_API_IMAGE"
docker tag "$REMOTE_WEB_IMAGE" "$LOCAL_WEB_IMAGE"

echo "Pulled and retagged images:"
echo "  $REMOTE_API_IMAGE -> $LOCAL_API_IMAGE"
echo "  $REMOTE_WEB_IMAGE -> $LOCAL_WEB_IMAGE"
