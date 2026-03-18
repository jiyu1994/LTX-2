#!/usr/bin/env bash
# =============================================================================
# LTX-2 Video-Only LoRA Training Script
# Uses GPUs 4,5,6,7 (4-way DDP), same dataset as diffsynth-studio Wan training.
# =============================================================================
set -euo pipefail

export CUDA_VISIBLE_DEVICES=4,5,6,7

# ─── Paths (modify these) ───────────────────────────────────────────────────
# LTX-2 model checkpoint
MODEL_PATH="/path/to/ltx-2-19b-dev.safetensors"
# Gemma text encoder directory
TEXT_ENCODER_PATH="/path/to/gemma-3-12b-it-qat-q4_0-unquantized"
# diffsynth-studio Wan dataset root (contains videos/ and images/ subdirs)
WAN_DATASET_ROOT="/path/to/diffsynth-studio/14b_lora/dataset_wan_real"
# diffsynth-studio metadata JSON (after clean)
WAN_METADATA_JSON="/path/to/diffsynth-studio/14b_lora/dataset_wan_real/metadata_clean.json"
# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUN_TS="$(date "+%Y%m%d_%H%M%S")"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/ltx2_video_lora_${RUN_TS}"
CONVERTED_JSON="${WAN_DATASET_ROOT}/dataset_ltx2.json"
PRECOMPUTED_DIR="${WAN_DATASET_ROOT}/.precomputed"
RESOLUTION_BUCKETS="832x480x81"
NUM_GPUS=4

timestamp() { date "+%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }

# ─── Step 1: Convert dataset format ─────────────────────────────────────────
if [[ ! -f "${CONVERTED_JSON}" ]]; then
  log "Step 1/3: Converting Wan dataset metadata to LTX-2 format..."
  cd "${SCRIPT_DIR}"
  uv run python scripts/convert_wan_dataset.py \
    "${WAN_METADATA_JSON}" \
    --dataset-root "${WAN_DATASET_ROOT}" \
    --output "${CONVERTED_JSON}"
  log "Dataset converted: ${CONVERTED_JSON}"
else
  log "Step 1/3: Converted dataset already exists, skipping: ${CONVERTED_JSON}"
fi

# ─── Step 2: Preprocess dataset (compute latents & embeddings) ──────────────
if [[ ! -d "${PRECOMPUTED_DIR}/latents" ]] || [[ ! -d "${PRECOMPUTED_DIR}/conditions" ]]; then
  log "Step 2/3: Preprocessing dataset (computing latents & text embeddings)..."
  cd "${SCRIPT_DIR}"
  uv run python scripts/process_dataset.py "${CONVERTED_JSON}" \
    --resolution-buckets "${RESOLUTION_BUCKETS}" \
    --model-path "${MODEL_PATH}" \
    --text-encoder-path "${TEXT_ENCODER_PATH}"
  log "Preprocessing complete: ${PRECOMPUTED_DIR}"
else
  log "Step 2/3: Preprocessed data already exists, skipping: ${PRECOMPUTED_DIR}"
fi

# ─── Step 3: Generate runtime config & launch training ──────────────────────
log "Step 3/3: Starting LoRA training on GPUs ${CUDA_VISIBLE_DEVICES}..."
log "Output dir: ${OUTPUT_DIR}"

RUNTIME_CONFIG="${OUTPUT_DIR}/training_config.yaml"
mkdir -p "${OUTPUT_DIR}"

cat > "${RUNTIME_CONFIG}" <<YAML
model:
  model_path: "${MODEL_PATH}"
  text_encoder_path: "${TEXT_ENCODER_PATH}"
  training_mode: "lora"
  load_checkpoint: null

lora:
  rank: 32
  alpha: 32
  dropout: 0.0
  target_modules:
    - "attn1.to_k"
    - "attn1.to_q"
    - "attn1.to_v"
    - "attn1.to_out.0"
    - "attn2.to_k"
    - "attn2.to_q"
    - "attn2.to_v"
    - "attn2.to_out.0"
    - "ff.net.0.proj"
    - "ff.net.2"

training_strategy:
  name: "text_to_video"
  first_frame_conditioning_p: 0.5
  with_audio: false

optimization:
  learning_rate: 1e-4
  steps: 2000
  batch_size: 1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  optimizer_type: "adamw"
  scheduler_type: "linear"
  scheduler_params: {}
  enable_gradient_checkpointing: true

acceleration:
  mixed_precision_mode: "bf16"
  quantization: null
  load_text_encoder_in_8bit: false

data:
  preprocessed_data_root: "${PRECOMPUTED_DIR}"
  num_dataloader_workers: 4

validation:
  prompts:
    - "A woman with long brown hair sits at a wooden desk in a cozy home office, typing on a laptop while occasionally glancing at notes beside her. Soft natural light streams through a large window, casting warm shadows across the room."
    - "A chef in a white uniform stands in a professional kitchen, carefully plating a gourmet dish with precise movements. Steam rises from freshly cooked vegetables as he arranges them with tweezers."
  negative_prompt: "worst quality, inconsistent motion, blurry, jittery, distorted"
  images: null
  video_dims: [832, 480, 81]
  frame_rate: 25.0
  seed: 42
  inference_steps: 30
  interval: 200
  videos_per_prompt: 1
  guidance_scale: 4.0
  stg_scale: 1.0
  stg_blocks: [29]
  stg_mode: "stg_v"
  generate_audio: false
  skip_initial_validation: false

checkpoints:
  interval: 500
  keep_last_n: -1
  precision: "bfloat16"

flow_matching:
  timestep_sampling_mode: "shifted_logit_normal"
  timestep_sampling_params: {}

hub:
  push_to_hub: false
  hub_model_id: null

wandb:
  enabled: false
  project: "ltx-2-trainer"
  entity: null
  tags: ["ltx2", "video-lora"]
  log_validation_videos: true

seed: 42
output_dir: "${OUTPUT_DIR}"
YAML

log "Runtime config written: ${RUNTIME_CONFIG}"

cd "${SCRIPT_DIR}"
uv run accelerate launch \
  --config_file configs/accelerate/ddp.yaml \
  --num_processes ${NUM_GPUS} \
  scripts/train.py "${RUNTIME_CONFIG}"

log "Training finished. Output: ${OUTPUT_DIR}"
