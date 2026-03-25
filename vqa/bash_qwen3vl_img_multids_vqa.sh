#!/bin/bash
set -euo pipefail

#############################################
# Global config shared by all datasets
#############################################
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

BATCH_SIZE=16
NUM_SAMPLES=2000000
MAX_NEW_TOKENS=500
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.05
MIN_PIXELS=$((256*32*32))
MAX_PIXELS=$((384*32*32))
DTYPE="bfloat16"
FLASH_ATTN2=0
DEVICE_MAP="cuda"

MAX_JOBS=1

#############################################
# Dataset pairs: add as many as you want
#############################################
IMAGE_DIRS=(
  "/4tb/dataset/social_media/HOD/class"
  "/4tb/dataset/social_media/facebook-hateful-meme-dataset"
  "/4tb/dataset/social_media/MultiOFF"
)

OUTPUT_DIRS=(
  "/4tb/dataset/social_media/HOD/Qwen3-VL/output/version_4"
  "/4tb/dataset/social_media/facebook-hateful-meme-dataset/Qwen3-VL/output/version_4"
  "/4tb/dataset/social_media/MultiOFF/Qwen3-VL/output/version_4"
)

#############################################
# Questions (Option 1: explicit \n newlines)
#############################################
QUESTIONS=(
  "Does the image contain violent, hateful, unsafe, 18+, or any other offensive content (e.g. alcohol, weapons, blood, cigarette, insulting gestures) or texts? Answer Yes or No. If Yes, specify the category clearly."
  "Rate the overall toxicity level of the image on a continuous scale from 0.0 to 1.0, where 0.0 means completely safe or non-toxic, and 1.0 means extremely toxic or unsafe.\nToxicity includes explicit, offensive, violent, hateful, unsafe, 18+, or otherwise inappropriate content such as alcohol, weapons, blood, cigarettes, or insulting gestures, as well as toxic text content.\nIf no such elements appear, give 0.0.\nProvide the numeric score and a short justification."
)

#############################################
# Function to run one dataset
#############################################
run_vqa() {
  local image_dir="$1"
  local output_dir="$2"

  echo "==============================================="
  echo "[START] image_dir: $image_dir"
  echo "[START] output_dir: $output_dir"
  echo "==============================================="

  mkdir -p "$output_dir"

  python /home/labad/minxing/code/Qwen3-VL/vqa/qwen3vl_img_vqa.py \
    --image_dir "$image_dir" \
    --output_dir "$output_dir" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --dtype "$DTYPE" \
    --device_map "$DEVICE_MAP" \
    ${MIN_PIXELS:+--min_pixels "$MIN_PIXELS"} \
    ${MAX_PIXELS:+--max_pixels "$MAX_PIXELS"} \
    $( [[ -n "${NUM_SAMPLES:-}" ]] && echo --num_samples "$NUM_SAMPLES" ) \
    $( [[ "$FLASH_ATTN2" == "1" ]] && echo --flash_attn2 ) \
    --questions "${QUESTIONS[@]}"

  echo "[DONE ] $image_dir -> $output_dir"
}

#############################################
# Sanity checks
#############################################
if [[ ${#IMAGE_DIRS[@]} -ne ${#OUTPUT_DIRS[@]} ]]; then
  echo "ERROR: IMAGE_DIRS and OUTPUT_DIRS must have the same length." >&2
  exit 1
fi

#############################################
# Scheduler (sequential or simple parallel)
#############################################
running=0
pids=()

for idx in "${!IMAGE_DIRS[@]}"; do
  img="${IMAGE_DIRS[$idx]}"
  out="${OUTPUT_DIRS[$idx]}"

  run_vqa "$img" "$out" & pids+=($!)
  running=$((running+1))

  if (( running >= MAX_JOBS )); then
    wait -n
    running=$((running-1))
  fi
done

wait "${pids[@]/#/}" || true

echo "==============================================="
echo "All dataset jobs completed."
echo "==============================================="
