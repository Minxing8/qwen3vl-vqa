#!/bin/bash
set -e

VIDEO_DIR="/path/to/videos_root"
OUTPUT_DIR="/path/to/output/qwen3vl_vid_vqa"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

FPS=2.0
# NFRAMES=64

TOTAL_PIXELS=$((20480*32*32))
MIN_PIXELS=$((16*32*32))
# MAX_PIXELS=$((768*32*32))

MAX_NEW_TOKENS=256
TEMPERATURE=0.3
TOP_P=0.9
TOP_K=20
REPETITION_PENALTY=1.05
NUM_SAMPLES=
DTYPE="bfloat16"
FLASH_ATTN2=0
DEVICE_MAP="auto"

QUESTIONS=(
  "Describe the video."
  "List key events with timestamps if visible."
  "Identify notable objects and their counts."
)

python /home/labad/minxing/code/Qwen3-VL/vqa/qwen3vl_vid_vqa.py \
  --video_dir "$VIDEO_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --top_k $TOP_K \
  --repetition_penalty $REPETITION_PENALTY \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  --total_pixels $TOTAL_PIXELS \
  --min_pixels $MIN_PIXELS \
  $( [[ -n "$MAX_PIXELS" ]] && echo --max_pixels $MAX_PIXELS ) \
  $( [[ -n "$FPS" ]] && echo --fps $FPS ) \
  $( [[ -n "$NFRAMES" ]] && echo --nframes $NFRAMES ) \
  $( [[ -n "$NUM_SAMPLES" ]] && echo --num_samples $NUM_SAMPLES ) \
  $( [[ "$FLASH_ATTN2" == "1" ]] && echo --flash_attn2 ) \
  --questions "${QUESTIONS[@]}"
