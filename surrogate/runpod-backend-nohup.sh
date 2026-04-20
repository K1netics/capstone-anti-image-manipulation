#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${PHOTOGUARD_LOG_DIR:-${ROOT_DIR}/logs}"
LOG_FILE="${PHOTOGUARD_LOG_FILE:-${LOG_DIR}/backend.log}"
PID_FILE="${PHOTOGUARD_PID_FILE:-${LOG_DIR}/backend.pid}"

mkdir -p "${LOG_DIR}"

nohup bash "${ROOT_DIR}/runpod-backend.sh" > "${LOG_FILE}" 2>&1 &
PID="$!"
echo "${PID}" > "${PID_FILE}"

echo "backend_pid=${PID}"
echo "log_file=${LOG_FILE}"
echo "pid_file=${PID_FILE}"
