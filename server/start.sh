#!/bin/bash
# =============================================================================
# LTX-2 推理服务启动脚本
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

HOST="${LTX_HOST:-0.0.0.0}"
PORT="${LTX_PORT:-60317}"
WORKERS="${LTX_WORKERS:-1}"

echo "============================================"
echo " LTX-2 Video Generation API Server"
echo "============================================"
echo " Host:     ${HOST}"
echo " Port:     ${PORT}"
echo " Config:   ${SCRIPT_DIR}/config.yaml"
echo " API Docs: http://${HOST}:${PORT}/docs"
echo "============================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec uv run uvicorn server.app:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --timeout-keep-alive 300
