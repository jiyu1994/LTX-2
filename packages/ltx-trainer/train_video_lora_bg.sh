#!/usr/bin/env bash
# =============================================================================
# Background launcher for LTX-2 video LoRA training.
# Mirrors diffsynth-studio/14b_lora/train_bg.sh behaviour:
#   - Prevents duplicate runs
#   - Logs to a timestamped file
#   - Records PID for stop_train_bg.sh
# =============================================================================
set -euo pipefail

timestamp() { date "+%Y%m%d_%H%M%S"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_video_lora.sh"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${LOG_DIR}"

RUN_NAME="train_$(timestamp)"
LOG_FILE="${LOG_DIR}/${RUN_NAME}.log"
PID_FILE="${LOG_DIR}/${RUN_NAME}.pid"
LATEST_PID_FILE="${LOG_DIR}/latest_train.pid"

if [[ -f "${LATEST_PID_FILE}" ]]; then
  OLD_PID="$(tr -d '[:space:]' < "${LATEST_PID_FILE}")"
  if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
    echo "A training process is already running: PID=${OLD_PID}"
    echo "Stop it first, then retry."
    exit 1
  fi
fi

echo "Starting LTX-2 video LoRA training in background..."
echo "Train script : ${TRAIN_SCRIPT}"
echo "Log file     : ${LOG_FILE}"

nohup setsid bash "${TRAIN_SCRIPT}" > "${LOG_FILE}" 2>&1 < /dev/null &
PID=$!

echo "${PID}" > "${PID_FILE}"
echo "${PID}" > "${LATEST_PID_FILE}"

echo "Started. PID=${PID}"
echo "PID file     : ${PID_FILE}"
echo "Latest PID   : ${LATEST_PID_FILE}"
echo "Watch log    : tail -f ${LOG_FILE}"
