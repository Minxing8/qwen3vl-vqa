#!/bin/bash
set -e

# ===== Adjust these =====
# IMAGE_DIR="/proj/berzelius-2024-90/users/datasets/mmreid/Market-1501-v15.09.15"
IMAGE_DIR="/proj/berzelius-2024-90/users/x_liumi/Qwen2.5-VL/dataset/24_1/media_files/2024-01-01_2024-03-01"
# OUTPUT_DIR="/proj/berzelius-2024-90/users/x_liumi/Qwen3-VL/output/market1501"
OUTPUT_DIR="/proj/berzelius-2024-90/users/x_liumi/Qwen3-VL/output/Qwen3-VL/output/version_4/split_1"
MODEL_NAME="Qwen/Qwen3-VL-8B-Instruct"

NUM_GPUS=8
NNODES=1
RDZV_ENDPOINT="localhost:29500"

BATCH_SIZE=16
NUM_SAMPLES=200000
SHUFFLE=1
SEED=2025

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
  # ReID
  # "Describe the person’s appearance for person re-identification. Focus only on clothing, colors, patterns, shoes, bags or backpacks, hats, hair, and other worn or carried items. Do NOT mention background, scene, camera view, or location. Output 1–2 concise sentences."

  # Social media
  # "Does this image contain any visible text? Answer Yes or No. If Yes, transcribe the text exactly as it appears without paraphrasing or interpretation." # OCR
  # "Does the image contain one or more humans? Answer Yes or No.\nIf Yes, for each visible person, provide a separate description including:\n- Person number (e.g., Person 1, Person 2, ...),\n- Name if recognizable,\n- Gender,\n- Approximate age,\n- Outfit,\n- Activity or pose,\n- Facial expression,\n- Interactions with other people (if any).\nKeep the description concise and factual, avoiding assumptions." # Human
  # "Provide a concise description of the image in 2–3 sentences, mentioning the key objects, people, actions, background, texts, and overall scene." # General
  # "Does the image contain violent, hateful, unsafe, 18+, or any other offensive content (e.g. alcohol, weapons, blood, cigarette, insulting gestures) or texts? Answer Yes or No. If Yes, specify the category clearly." # Toxicity
  # "What is the main theme of the image? Choose the most specific category that fits best from: politics, entertainment, sports, technology, anime, gaming, fashion, beauty/makeup, photography, food, pets/animals, nature, travel, memes, advertising, daily life, art, health/fitness, science, religion, history, news/crime. Avoid 'other' unless nothing else fits. Provide a short justification." # Topic
  # "If the image contains one or more humans, rate each visible person’s facial and/or body attractiveness on a scale of 1–10, where 1 = least attractive and 10 = most attractive, based on general visual aesthetics, symmetry, and presentation.\nIf no humans are visible, answer 'No human detected.'\nProvide your answer in the format:\nPerson 1: [score]; Person 2: [score]; ...\nOptionally include a one-sentence justification." # Attractiveness (1–10)
  # "If the image contains one or more humans, classify each visible person’s facial and/or body attractiveness into one of three levels: 'Low', 'Medium', or 'High', based on general aesthetic impression, symmetry, and presentation.\nIf no humans are visible, answer 'No human detected.'\nProvide your answer in the format:\nPerson 1: [level]; Person 2: [level];\nOptionally include a short justification." # Attractiveness (Low/Med/High)
  # "Rate the overall toxicity level of the image on a continuous scale from 0.0 to 1.0, where 0.0 means completely safe or non-toxic, and 1.0 means extremely toxic or unsafe.\nToxicity includes explicit, offensive, violent, hateful, unsafe, 18+, or otherwise inappropriate content such as alcohol, weapons, blood, cigarettes, or insulting gestures, as well as toxic text content.\nIf no such elements appear, give 0.0.\nProvide the numeric score and a short justification." # Toxicity (score)

  # Social media version 5
  # Type & Quality & Aeththetics
  "Classify the image format into exactly one label using the definitions below. Return the label name only (not the description). Formats (label: description): photo — real-world camera photo of people, places, or objects (not a UI capture). screenshot — capture of an app or website showing UI elements (menus, buttons, bars). news_card — designed news graphic, broadcast overlay, weather alert, or security footage presenting a headline (not a screenshot). flyer_poster — event or announcement layout with date, time, location, or call-to-action. advertisement_graphic — promotional graphic for a product, brand, or service. infographic — visual explanation using charts, statistics, or structured facts. document — scanned or photographed document, form, or letter. painting — hand-drawn, painted, sketched, illustrated, or comic-style artwork. meme — humorous or reaction-style image using a meme format. other — none of the above formats clearly apply. If multiple formats seem possible, choose the closest single match. Use \"other\" only if no format reasonably fits. Also provide two 1–5 ratings:  Aesthetic score (visual appeal/composition, independent of resolution): 1 — very poor: cluttered, awkward layout, unpleasant visual balance. 2 — below average: weak composition or distracting arrangement. 3 — average: acceptable, clear, but not visually strong. 4 — good: well-composed, balanced, visually pleasing. 5 — excellent: highly appealing, polished, professional-looking. Quality score (technical image quality: sharpness, resolution, noise, artifacts): 1 — very poor: blurry/low-res, heavy artifacts, hard to view. 2 — below average: noticeable blur/low-res or artifacts. 3 — average: usable and readable, minor flaws. 4 — good: sharp, clear, minimal artifacts. 5 — excellent: very sharp, high-res, clean and well-exposed. Assign a confidence score between 0 and 1 indicating how confident you are that the image_type classification is correct. Return valid JSON only: {  \"image_type\": \"<one label>\",  \"aesthetic_score\": number,  \"quality_score\": number,  \"confidence\": number } Do not return multiple labels or include any text outside the JSON.",
  # OCR
  "You are an OCR and layout extraction model. Process the image in two distinct phases: Phase 1: Raw Transcription Detect and transcribe: EVERY piece of visible text in the image exactly as it appears. Include everything: headlines, body text, logos, brand names, and watermarks. Do not omit anything. Preserve capitalization and line breaks. Phase 2: Extract the text based on these rules: 1. Headline: The most prominent large text. 2. Subtitle: The smaller supporting text directly associated with the headline. 3. Exclusion Rule: In this phase ONLY, discard all brand names, logos (e.g., \"12NEWS\"), URLs, and watermarks. Output Format: Output the result as valid JSON only: { \"has_text\": boolean, \"transcription\": \"String containing EVERY word from Phase 1\", \"headline\": \"String or null\", \"subtitle\": \"String or null\", \"confidence\": float }",
  # Topic
  "Choose the primary topic of the image. Pick exactly ONE topic_id from the list below and return the topic_id only (not the description text). Use \"other\" ONLY if none of the listed topics clearly apply. Topics (topic_id: description): politics: government, elections, public policy, political actors. crime: criminal acts, violence, investigations, arrests. natural_disaster: earthquakes, floods, fires, storms, natural damage. entertainment: celebrities, movies, music, TV, pop culture. sports: athletic events, teams, players, competitions. technology: software, hardware, AI, digital products. science: research, discoveries, space, experiments. health: medicine, disease, public health, wellness. fitness: exercise, workouts, physical training. gaming: video games, esports, gaming culture. anime: anime or manga-related content. fashion: clothing, style, accessories. beauty_makeup: cosmetics, skincare, makeup, appearance. food: cooking, meals, restaurants. pets_animals: animals or pets as the main subject. nature: landscapes, wildlife, natural scenes. travel: tourism, destinations, travel experiences. art: creative artwork or artistic expression. photography: photography as a subject (gear, technique, portfolios). religion: religious beliefs, rituals, symbols. history: historical events figures or eras memes humorous or reaction-style meme content advertising promotional or marketing-focused content daily_life everyday personal moments or routine activities other none of the above topics clearly apply If multiple topics seem possible choose the closest single match Only use \"other\" if no category reasonably fits Also output a confidence score from 0 to 1 and a one-sentence justification Return JSON only {  \"topic\": \"<topic_id>\",  \"confidence\": number  \"justification\": string} Do not include any text outside the JSON.",
  # Image Description
  "Describe the image using short, factual phrases. Only describe what is clearly visible and avoid assumptions. Fields: - people: visible people or their appearance. - objects: key visible objects. - action: what is happening. - background: setting or environment. - visible_text: any clearly readable text. Use at most 12 words per field. If a field is not applicable, set it to null. Assign a confidence score between 0 and 1 indicating how confident you are that this description accurately represents the image. Use high confidence only if the scene is clear and unambiguous. Return valid JSON only: { \"people\": string | null, \"objects\": string | null, \"action\": string | null, \"background\": string | null, \"visible_text\": string | null, \"confidence\": number } Do not include any text outside the JSON object.",
)

SCRIPT="/proj/berzelius-2024-409/users/x_liumi/Qwen3-VL/vqa/qwen3vl_img_vqa_parallel.py"

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
