#!/usr/bin/env bash
set -euo pipefail

API_CONTAINER="${API_CONTAINER:-photoguard-api}"
WEB_CONTAINER="${WEB_CONTAINER:-photoguard-web}"

docker rm -f "$WEB_CONTAINER" "$API_CONTAINER" >/dev/null 2>&1 || true

echo "Stopped containers:"
echo "  $API_CONTAINER"
echo "  $WEB_CONTAINER"
