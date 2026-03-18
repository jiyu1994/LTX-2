#!/usr/bin/env bash
# Stop the background LTX-2 training process launched by train_video_lora_bg.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LATEST_PID_FILE="${SCRIPT_DIR}/logs/latest_train.pid"

if [[ ! -f "${LATEST_PID_FILE}" ]]; then
  echo "No PID file found: ${LATEST_PID_FILE}"
  exit 0
fi

PID="$(tr -d '[:space:]' < "${LATEST_PID_FILE}")"

if [[ -z "${PID}" ]]; then
  echo "PID file is empty."
  exit 0
fi

if kill -0 "${PID}" 2>/dev/null; then
  echo "Stopping training process group (PID=${PID})..."
  kill -- -"${PID}" 2>/dev/null || kill "${PID}" 2>/dev/null || true
  echo "Stopped."
else
  echo "Process ${PID} is not running."
fi
