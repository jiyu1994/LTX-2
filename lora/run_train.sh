#!/bin/bash
# =============================================================================
# LTX-2 Video-Only LoRA 全流程训练脚本
# =============================================================================
#
# 硬件: 4x H200 (GPU 4,5,6,7)
# 数据: ~400 样本 × 10 epochs = 1000 steps
#
# 使用前请修改下面的路径配置
# =============================================================================

set -euo pipefail

# ==================== 路径配置 (请修改) ====================

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL_PATH="/path/to/ltx-2-19b-dev.safetensors"
TEXT_ENCODER_PATH="/path/to/gemma-3-12b-it-qat-q4_0-unquantized"

DATASET_DIR="/path/to/civitaiNSFWVideoDataset_800"
METADATA_FILE="${DATASET_DIR}/train_metadata.jsonl"

PRECOMPUTED_DIR="${DATASET_DIR}/.precomputed"

TRAIN_CONFIG="${REPO_ROOT}/lora/train_video_lora.yaml"
ACCELERATE_CONFIG="${REPO_ROOT}/lora/accelerate_config.yaml"

RESOLUTION_BUCKETS="512x512x129"

LORA_TRIGGER=""

# ==================== Step 1: 预处理数据集 ====================

echo "============================================"
echo "Step 1: 预处理数据集 (视频编码 + 文本嵌入)"
echo "============================================"

if [ ! -f "${METADATA_FILE}" ]; then
    echo "错误: 未找到 ${METADATA_FILE}"
    echo "请先运行: python lora/data_prepare.py --resample"
    exit 1
fi

cd "${REPO_ROOT}/packages/ltx-trainer"

if [ -d "${PRECOMPUTED_DIR}/latents" ] && [ -d "${PRECOMPUTED_DIR}/conditions" ]; then
    echo "预处理数据已存在，跳过 (如需重新处理请删除 ${PRECOMPUTED_DIR})"
else
    PREPROCESS_CMD="uv run python scripts/process_dataset.py ${METADATA_FILE} \
        --resolution-buckets '${RESOLUTION_BUCKETS}' \
        --model-path ${MODEL_PATH} \
        --text-encoder-path ${TEXT_ENCODER_PATH} \
        --output-dir ${PRECOMPUTED_DIR}"

    if [ -n "${LORA_TRIGGER}" ]; then
        PREPROCESS_CMD="${PREPROCESS_CMD} --lora-trigger '${LORA_TRIGGER}'"
    fi

    echo "运行: ${PREPROCESS_CMD}"
    eval ${PREPROCESS_CMD}

    echo "预处理完成"
fi

# ==================== Step 2: 8x H200 DDP 训练 ====================

echo ""
echo "============================================"
echo "Step 2: 开始 4x H200 DDP LoRA 训练 (GPU 4,5,6,7)"
echo "============================================"

RUNTIME_CONFIG="/tmp/ltx2_train_config.yaml"
sed \
    -e "s|model_path:.*|model_path: \"${MODEL_PATH}\"|" \
    -e "s|text_encoder_path:.*|text_encoder_path: \"${TEXT_ENCODER_PATH}\"|" \
    -e "s|preprocessed_data_root:.*|preprocessed_data_root: \"${PRECOMPUTED_DIR}\"|" \
    "${TRAIN_CONFIG}" > "${RUNTIME_CONFIG}"

echo "训练配置: ${RUNTIME_CONFIG}"
echo "Accelerate 配置: ${ACCELERATE_CONFIG}"
echo ""
echo "  数据:   ~400 样本"
echo "  Epochs: 10"
echo "  GPU:    4x H200 (4,5,6,7)"
echo "  Batch:  1/GPU × 4 GPU = 4 effective"
echo "  Steps:  1000 (每 epoch 100 步)"
echo "  LoRA:   rank=64, 视频注意力层"
echo ""
echo "开始训练..."

uv run accelerate launch \
    --config_file "${ACCELERATE_CONFIG}" \
    scripts/train.py "${RUNTIME_CONFIG}"

echo ""
echo "============================================"
echo "训练完成!"
echo "============================================"
echo "Checkpoint 目录: $(grep output_dir ${RUNTIME_CONFIG} | awk '{print $2}' | tr -d '\"')"
echo "每 100 步 (每 epoch) 保存一个 checkpoint，共 10 个"
