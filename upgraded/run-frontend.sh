#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$ROOT_DIR/.tools/node/bin:$PATH"

cd "$ROOT_DIR"
exec npm run dev -- --host 127.0.0.1 --port 5173 "$@"
