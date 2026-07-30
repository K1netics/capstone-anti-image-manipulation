#!/bin/sh
set -eu

API_UPSTREAM_VALUE="${API_UPSTREAM:-/api}"

cat > /usr/share/nginx/html/runtime-config.js <<EOF
window.__IMMUNA_CONFIG__ = {
  API_UPSTREAM: "${API_UPSTREAM_VALUE}"
};
EOF
