#!/bin/bash
set -e

# ===== Adjust these =====
# IMAGE_DIR="/proj/berzelius-2024-90/users/datasets/mmreid/Market-1501-v15.09.15"
IMAGE_DIR="/4tb/dataset/social_media/ins/24_1/media_files/2024-01-01_2024-03-01"
# OUTPUT_DIR="/proj/berzelius-2024-90/users/x_liumi/Qwen3-VL/output/market1501_img_vqa_ddp_cleaned"
OUTPUT_DIR="/4tb/dataset/social_media/ins/24_1/Qwen3-VL/output/version_1"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

NUM_GPUS=8
NNODES=1
RDZV_ENDPOINT="localhost:29500"

BATCH_SIZE=64
NUM_SAMPLES=200000
SHUFFLE=1
SEED=2024

MIN_PIXELS=$((256*32*32))
MAX_PIXELS=$((1280*32*32))

MAX_NEW_TOKENS=500
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.05

DTYPE="bfloat16"
FLASH_ATTN2=0

QUESTIONS=(
    "One compact sentence (<=25 words) describing only stable physical traits: perceived sex; broad age band; height category; build; limb-torso proportion; head/face shape; hair length and color; posture/gait; and any distinctive marks with location. Use concrete, discriminative words; omit unknown traits. Do not mention clothing/logos/accessories or start with 'The person/individual,' and avoid the phrases 'appears,' 'likely,' 'average height/build,' 'proportionate,' and 'No visible...'. Return only the sentence."
    "In <=2 sentences, prioritize discriminative traits: (1) body build + approximate age band + height category (avoid 'average,' 'appears/likely'); (2) hair details + notable anatomical features (e.g., jawline, cheekbones, brow, facial hair, skin marks with location) + posture/gait/gesture. Mention clothing only if uniquely identifying, in <=5 words; otherwise omit. Vary phrasing; do not start with 'The person/individual.' Output only the description."
)

SCRIPT="/home/labad/minxing/code/Qwen3-VL/vqa/qwen3vl_img_vqa_parallel.py"

PIXEL_FLAGS=()
[[ -n "$MIN_PIXELS" ]] && PIXEL_FLAGS+=( --min_pixels "$MIN_PIXELS" )
[[ -n "$MAX_PIXELS" ]] && PIXEL_FLAGS+=( --max_pixels "$MAX_PIXELS" )

SHUFFLE_FLAG=()
[[ "$SHUFFLE" == "1" ]] && SHUFFLE_FLAG+=( --shuffle )

NUM_SAMPLES_FLAG=()
[[ -n "$NUM_SAMPLES" ]] && NUM_SAMPLES_FLAG+=( --num_samples "$NUM_SAMPLES" )

FA2_FLAG=()
[[ "$FLASH_ATTN2" == "1" ]] && FA2_FLAG+=( --flash_attn2 )

torchrun \
  --nproc_per_node="$NUM_GPUS" \
  --nnodes="$NNODES" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$RDZV_ENDPOINT" \
  "$SCRIPT" \
    --image_dir "$IMAGE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --dtype "$DTYPE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --top_k "$TOP_K" \
    --repetition_penalty "$REPETITION_PENALTY" \
    --seed "$SEED" \
    "${PIXEL_FLAGS[@]}" \
    "${SHUFFLE_FLAG[@]}" \
    "${NUM_SAMPLES_FLAG[@]}" \
    "${FA2_FLAG[@]}" \
    --questions "${QUESTIONS[@]}"
