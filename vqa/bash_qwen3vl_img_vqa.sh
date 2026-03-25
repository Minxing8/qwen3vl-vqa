#!/bin/bash
set -e

# IMAGE_DIR="/proj/berzelius-2024-90/users/datasets/mmreid/Market-1501-v15.09.15"
IMAGE_DIR="/4tb/dataset/social_media/ins/24_1/media_files/2024-01-01_2024-03-01"
# OUTPUT_DIR="/proj/berzelius-2024-90/users/x_liumi/Qwen3-VL/output/market1501_img_vqa"
OUTPUT_DIR="/4tb/dataset/social_media/ins/24_1/Qwen3-VL/output/version_4"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

BATCH_SIZE=16
NUM_SAMPLES=200000
MAX_NEW_TOKENS=500
TEMPERATURE=0.6
TOP_P=0.9
TOP_K=50
REPETITION_PENALTY=1.05
MIN_PIXELS=$((256*32*32))
MAX_PIXELS=$((384*32*32))
DTYPE="bfloat16"
FLASH_ATTN2=1
DEVICE_MAP="cuda"

QUESTIONS=(
  "Does this image contain any visible text? Answer Yes or No. If Yes, transcribe the text exactly as it appears without paraphrasing or interpretation."
  "Does the image contain one or more humans? Answer Yes or No.\nIf Yes, for each visible person, provide a separate description including:\n- Person number (e.g., Person 1, Person 2, ...),\n- Name if recognizable,\n- Gender,\n- Approximate age,\n- Outfit,\n- Activity or pose,\n- Facial expression,\n- Interactions with other people (if any).\nKeep the description concise and factual, avoiding assumptions."
  "Provide a concise description of the image in 2-3 sentences, mentioning the key objects, people, actions, background, texts, and overall scene."
  "Does the image contain violent, hateful, unsafe, 18+, or any other offensive content (e.g. alcohol, weapons, blood, cigarette, insulting gestures) or texts? Answer Yes or No. If Yes, specify the category clearly."
  "What is the main theme of the image? Choose the most specific category that fits best from: politics, entertainment, sports, technology, anime, gaming, fashion, beauty/makeup, photography, food, pets/animals, nature, travel, memes, advertising, daily life, art, health/fitness, science, religion, history, news/crime. Avoid 'other' unless nothing else fits. Provide a short justification."
  "If the image contains one or more humans, rate each visible person's facial and/or body attractiveness on a scale of 1-10, where 1 = least attractive and 10 = most attractive, based on general visual aesthetics, symmetry, and presentation.\nIf no humans are visible, answer 'No human detected.'\nProvide your answer in the format:\nPerson 1: [score]; Person 2: [score]; ...\nOptionally include a one-sentence justification."
  "If the image contains one or more humans, classify each visible person's facial and/or body attractiveness into one of three levels: 'Low', 'Medium', or 'High', based on general aesthetic impression, symmetry, and presentation.\nIf no humans are visible, answer 'No human detected.'\nProvide your answer in the format:\nPerson 1: [level]; Person 2: [level];\nOptionally include a short justification."
  "Rate the overall toxicity level of the image on a continuous scale from 0.0 to 1.0, where 0.0 means completely safe or non-toxic, and 1.0 means extremely toxic or unsafe.\nToxicity includes explicit, offensive, violent, hateful, unsafe, 18+, or otherwise inappropriate content such as alcohol, weapons, blood, cigarettes, or insulting gestures, as well as toxic text content.\nIf no such elements appear, give 0.0.\nProvide the numeric score and a short justification."
)

python /home/labad/minxing/code/Qwen3-VL/vqa/qwen3vl_img_vqa.py \
  --image_dir "$IMAGE_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --model_name "$MODEL_NAME" \
  --batch_size $BATCH_SIZE \
  --max_new_tokens $MAX_NEW_TOKENS \
  --temperature $TEMPERATURE \
  --top_p $TOP_P \
  --top_k $TOP_K \
  --repetition_penalty $REPETITION_PENALTY \
  --dtype "$DTYPE" \
  --device_map "$DEVICE_MAP" \
  ${MIN_PIXELS:+--min_pixels $MIN_PIXELS} \
  ${MAX_PIXELS:+--max_pixels $MAX_PIXELS} \
  $( [[ -n "$NUM_SAMPLES" ]] && echo --num_samples $NUM_SAMPLES ) \
  $( [[ "$FLASH_ATTN2" == "1" ]] && echo --flash_attn2 ) \
  --questions "${QUESTIONS[@]}"
