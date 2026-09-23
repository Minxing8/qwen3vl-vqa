#!/bin/bash
set -eo pipefail

# Influencer_22 main-job yes/no prompt (Q12), run with vqa/main_job.py.
# The owner of each image is read from IMAGE_DIR/<handle>/..., so IMAGE_DIR must be
# the influencer_22 "images" folder, and INFLUENCER_LABEL_JSON must cover every handle.
# Single-process, one GPU.

# ===== Adjust these =====
REPO_DIR="/proj/berzelius-2024-409/users/x_liumi/Qwen3-VL"
IMAGE_DIR="/proj/berzelius-2024-409/users/dataset/influencer_22/images"
OUTPUT_DIR="/proj/berzelius-2024-409/users/x_liumi/Qwen3-VL/output/Qwen3-VL/output/influencer_22_v3"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

INFLUENCER_LABEL_JSON="$REPO_DIR/data_influencer22/influencer_label.json"

GPU=0

BATCH_SIZE=4
# Empty = use every image under IMAGE_DIR
NUM_SAMPLES=""
DTYPE="bfloat16"

# Match the local pilot: MIN/MAX unset
MIN_PIXELS=""
MAX_PIXELS=""

EXTRA_FLAGS=()
[[ -n "$NUM_SAMPLES" ]] && EXTRA_FLAGS+=( --num_samples "$NUM_SAMPLES" )
[[ -n "$MIN_PIXELS" ]] && EXTRA_FLAGS+=( --min_pixels "$MIN_PIXELS" )
[[ -n "$MAX_PIXELS" ]] && EXTRA_FLAGS+=( --max_pixels "$MAX_PIXELS" )

mkdir -p "$OUTPUT_DIR"
cd "$REPO_DIR/vqa"

# Output: $OUTPUT_DIR/<basename of OUTPUT_DIR>_q12.csv
CUDA_VISIBLE_DEVICES="$GPU" python main_job.py \
  --image_dir "$IMAGE_DIR" \
  --csv_output_path "$OUTPUT_DIR" \
  --question_index 12 \
  --influencer_label_json "$INFLUENCER_LABEL_JSON" \
  --model_name "$MODEL_NAME" \
  --batch_size "$BATCH_SIZE" \
  --dtype "$DTYPE" \
  "${EXTRA_FLAGS[@]}" \
  2>&1 | tee "$OUTPUT_DIR/main_job.log"
