#!/bin/bash
# =============================================================================
# LTX-2 推理运行脚本
# =============================================================================

# ---- 模型路径配置 ----
MODEL_ROOT="/home/yuji/.cache/modelscope/hub/models/Lightricks/LTX-2.3"

CHECKPOINT="${MODEL_ROOT}/ltx-2.3-22b-dev.safetensors"
DISTILLED_CHECKPOINT="${MODEL_ROOT}/ltx-2.3-22b-distilled.safetensors"
DISTILLED_LORA="${MODEL_ROOT}/ltx-2.3-22b-distilled-lora-384.safetensors"
SPATIAL_UPSAMPLER="${MODEL_ROOT}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
SPATIAL_UPSAMPLER_V11="${MODEL_ROOT}/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
SPATIAL_UPSAMPLER_1_5X="${MODEL_ROOT}/ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors"
TEMPORAL_UPSAMPLER="${MODEL_ROOT}/ltx-2.3-temporal-upscaler-x2-1.0.safetensors"

# Gemma 文本编码器路径（需要单独下载: google/gemma-3-12b-it-qat-q4_0-unquantized）
# 从 ModelScope 下载: modelscope download --model google/gemma-3-12b-it-qat-q4_0-unquantized --local_dir ./gemma-3-12b
GEMMA_ROOT="/home/yuji/.cache/modelscope/hub/models/google/gemma-3-12b-it-qat-q4_0-unquantized"

# ---- 输出配置 ----
OUTPUT_DIR="./outputs"
mkdir -p "${OUTPUT_DIR}"

# ---- 通用提示词 ----
PROMPT="A beautiful sunset over the ocean, golden light reflecting on gentle waves, seabirds flying across the sky"

# =============================================================================
# 1. 两阶段文本转视频（推荐，生产级质量）
# =============================================================================
run_ti2vid_two_stages() {
    echo "=== 运行: TI2VidTwoStagesPipeline ==="
    python -m ltx_pipelines.ti2vid_two_stages \
        --checkpoint-path "${CHECKPOINT}" \
        --distilled-lora "${DISTILLED_LORA}" 0.8 \
        --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
        --gemma-root "${GEMMA_ROOT}" \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/ti2vid_two_stages.mp4" \
        --num-frames 121 \
        --height 1024 \
        --width 1536
}

# =============================================================================
# 2. 两阶段 HQ 文本转视频（更高质量，更少步数）
# =============================================================================
run_ti2vid_two_stages_hq() {
    echo "=== 运行: TI2VidTwoStagesHQPipeline ==="
    python -m ltx_pipelines.ti2vid_two_stages_hq \
        --checkpoint-path "${CHECKPOINT}" \
        --distilled-lora "${DISTILLED_LORA}" 0.8 \
        --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
        --gemma-root "${GEMMA_ROOT}" \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/ti2vid_two_stages_hq.mp4" \
        --num-frames 121 \
        --num-inference-steps 15 \
        --height 1088 \
        --width 1920
}

# =============================================================================
# 3. 单阶段文本转视频（快速原型验证）
# =============================================================================
run_ti2vid_one_stage() {
    echo "=== 运行: TI2VidOneStagePipeline ==="
    python -m ltx_pipelines.ti2vid_one_stage \
        --checkpoint-path "${CHECKPOINT}" \
        --gemma-root "${GEMMA_ROOT}" \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/ti2vid_one_stage.mp4" \
        --num-frames 121 \
        --height 512 \
        --width 768
}

# =============================================================================
# 4. 蒸馏快速推理（最快，8步生成）
# =============================================================================
run_distilled() {
    echo "=== 运行: DistilledPipeline ==="
    python -m ltx_pipelines.distilled \
        --distilled-checkpoint-path "${DISTILLED_CHECKPOINT}" \
        --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
        --gemma-root "${GEMMA_ROOT}" \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/distilled.mp4" \
        --num-frames 121 \
        --height 1024 \
        --width 1536
}

# =============================================================================
# 5. IC-LoRA 视频转视频
# 需要额外下载 IC-LoRA 权重，例如:
#   IC_LORA_PATH="path/to/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
# =============================================================================
run_ic_lora() {
    local IC_LORA_PATH="${1:?请提供 IC-LoRA 权重路径}"
    local REF_VIDEO="${2:?请提供参考视频路径}"
    echo "=== 运行: ICLoraPipeline ==="
    python -m ltx_pipelines.ic_lora \
        --distilled-checkpoint-path "${DISTILLED_CHECKPOINT}" \
        --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
        --gemma-root "${GEMMA_ROOT}" \
        --lora "${IC_LORA_PATH}" 1.0 \
        --video-conditioning "${REF_VIDEO}" 1.0 \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/ic_lora.mp4" \
        --num-frames 121 \
        --height 1024 \
        --width 1536
}

# =============================================================================
# 6. 关键帧插值
# =============================================================================
run_keyframe_interpolation() {
    local IMAGE1="${1:?请提供第一个关键帧图像路径}"
    local IMAGE2="${2:?请提供第二个关键帧图像路径}"
    echo "=== 运行: KeyframeInterpolationPipeline ==="
    python -m ltx_pipelines.keyframe_interpolation \
        --checkpoint-path "${CHECKPOINT}" \
        --distilled-lora "${DISTILLED_LORA}" 0.8 \
        --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
        --gemma-root "${GEMMA_ROOT}" \
        --image "${IMAGE1}" 0 1.0 \
        --image "${IMAGE2}" 120 1.0 \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/keyframe_interpolation.mp4" \
        --num-frames 121 \
        --height 1024 \
        --width 1536
}

# =============================================================================
# 7. 音频驱动视频生成
# =============================================================================
run_a2vid() {
    local AUDIO_PATH="${1:?请提供音频文件路径}"
    echo "=== 运行: A2VidPipelineTwoStage ==="
    python -m ltx_pipelines.a2vid_two_stage \
        --checkpoint-path "${CHECKPOINT}" \
        --distilled-lora "${DISTILLED_LORA}" 0.8 \
        --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
        --gemma-root "${GEMMA_ROOT}" \
        --audio-path "${AUDIO_PATH}" \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/a2vid.mp4" \
        --num-frames 121 \
        --height 1024 \
        --width 1536
}

# =============================================================================
# 8. 视频局部重生成
# =============================================================================
run_retake() {
    local VIDEO_PATH="${1:?请提供源视频路径}"
    local START_TIME="${2:-1.0}"
    local END_TIME="${3:-3.0}"
    echo "=== 运行: RetakePipeline ==="
    python -m ltx_pipelines.retake \
        --checkpoint-path "${CHECKPOINT}" \
        --gemma-root "${GEMMA_ROOT}" \
        --video-path "${VIDEO_PATH}" \
        --start-time "${START_TIME}" \
        --end-time "${END_TIME}" \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/retake.mp4"
}

# =============================================================================
# FP8 量化版本（降低显存占用）
# 在任何管线前加上环境变量即可:
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 在命令中加上: --quantization fp8-cast
# =============================================================================
run_ti2vid_two_stages_fp8() {
    echo "=== 运行: TI2VidTwoStagesPipeline (FP8 量化) ==="
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -m ltx_pipelines.ti2vid_two_stages \
        --checkpoint-path "${CHECKPOINT}" \
        --distilled-lora "${DISTILLED_LORA}" 0.8 \
        --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
        --gemma-root "${GEMMA_ROOT}" \
        --quantization fp8-cast \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/ti2vid_two_stages_fp8.mp4" \
        --num-frames 121 \
        --height 1024 \
        --width 1536
}

# =============================================================================
# 图像转视频示例（在任何两阶段管线中加上 --image 参数）
# =============================================================================
run_image_to_video() {
    local IMAGE_PATH="${1:?请提供输入图像路径}"
    echo "=== 运行: Image-to-Video ==="
    python -m ltx_pipelines.ti2vid_two_stages \
        --checkpoint-path "${CHECKPOINT}" \
        --distilled-lora "${DISTILLED_LORA}" 0.8 \
        --spatial-upsampler-path "${SPATIAL_UPSAMPLER}" \
        --gemma-root "${GEMMA_ROOT}" \
        --image "${IMAGE_PATH}" 0 1.0 \
        --prompt "${PROMPT}" \
        --output-path "${OUTPUT_DIR}/image_to_video.mp4" \
        --num-frames 121 \
        --height 1024 \
        --width 1536
}

# =============================================================================
# 使用说明
# =============================================================================
usage() {
    echo ""
    echo "LTX-2 推理运行脚本"
    echo "=================================="
    echo ""
    echo "用法: source run.sh && <函数名> [参数]"
    echo ""
    echo "可用命令:"
    echo "  run_ti2vid_two_stages          - 两阶段文转视频（推荐）"
    echo "  run_ti2vid_two_stages_hq       - 两阶段 HQ 文转视频"
    echo "  run_ti2vid_one_stage           - 单阶段快速文转视频"
    echo "  run_distilled                  - 蒸馏快速推理（最快）"
    echo "  run_ic_lora <lora> <video>     - IC-LoRA 视频转视频"
    echo "  run_keyframe_interpolation <img1> <img2> - 关键帧插值"
    echo "  run_a2vid <audio>              - 音频驱动视频生成"
    echo "  run_retake <video> [start] [end] - 视频局部重生成"
    echo "  run_ti2vid_two_stages_fp8      - FP8 量化版（省显存）"
    echo "  run_image_to_video <image>     - 图像转视频"
    echo ""
    echo "示例:"
    echo "  source run.sh && run_ti2vid_two_stages"
    echo "  source run.sh && run_distilled"
    echo "  source run.sh && run_image_to_video /path/to/image.jpg"
    echo "  source run.sh && run_a2vid /path/to/audio.wav"
    echo ""
    echo "提示: 修改脚本顶部的 PROMPT 变量来更换提示词"
    echo ""
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    usage
fi
