#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from pillow_heif import register_heif_opener
from tqdm import tqdm

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

register_heif_opener()


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE_DIR = REPO_ROOT / "data_influencer22" / "sample_50"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data_influencer22" / "output" / "test_50" / "personal_brand"
DEFAULT_INFLUENCER_LABEL_PATH = REPO_ROOT / "data_influencer22" / "influencer_label.json"
DEFAULT_LABELS_TXT_PATH = REPO_ROOT / "data_influencer22" / "labels.txt"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}

PROMPT_PREFIX = (
    "Answer exactly one word: Yes or No. "
    "Do not identify or guess any person, influencer, account, brand, team, platform, or organization. "
    "Use only visible evidence such as clothing, actions, objects, scene type, readable text, and the image's apparent intent. "
    "Do not use outside knowledge."
)

PROMPTS_BY_LABEL = {
    "Reality TV Star": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly suggest a television or broadcast context, such as studio-ready styling, "
        "a microphone, speaking toward an audience or interviewer, or promotion of a TV show or channel?"
    ),
    "actor": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show a formal red-carpet or film-promotion context, or is it a film/series poster, trailer visual, "
        "screen capture, or on-set filming scene?"
    ),
    "basketball": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show basketball activity or promotion, such as a basketball game, warm-up, match promotion, "
        "a basketball jersey, or a visible basketball?"
    ),
    "comedian": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show comedy-performance or comedy-promotion cues, such as a microphone, stage or audience setup, "
        "speaking toward an audience or interviewer, or promotion of a comedy show or channel?"
    ),
    "content creator": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly contain humorous or playful content, such as a joke, prank, funny pose, exaggerated reaction, or comedic visual framing?"
    ),
    "cricket": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show cricket activity or promotion, such as a cricket match, match promotion, a cricket bat or ball, "
        "wickets, or cricket sportswear?"
    ),
    "entertainment company": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly relate to movie or comic entertainment, such as film promotion, a movie scene, a filming scene, "
        "a comic panel, or a comic-style character?"
    ),
    "fashion": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly emphasize fashion display or promotion, such as strong presentation of clothing, lingerie, accessories, bags, "
        "or a product-focused apparel image?"
    ),
    "football": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show football (soccer) activity or promotion, such as a match scene, warm-up, match promotion, "
        "a football kit, or a visible football?"
    ),
    "meme": (
        f"{PROMPT_PREFIX} "
        "Question: Is this image clearly a meme image or meme-style joke graphic?"
    ),
    "model": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show model-style self-presentation, such as body or styling display, heavy beauty or fashion emphasis, "
        "a runway context, an editorial or magazine-style shoot, or promotion of a fashion show?"
    ),
    "nature, photography": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show nature, wildlife, landscape, documentary-style human-interest photography, "
        "or educational content about nature, places, or people?"
    ),
    "politician": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show political activity, such as diplomacy, a public speech, a political meeting, campaign material, "
        "or promotion of political slogans or policies?"
    ),
    "singer": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show music-performance or music-promotion cues, such as singing, playing an instrument, "
        "wearing a performance outfit, or promotion of an album, concert, or live show?"
    ),
    "social media": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly promote or explain a social media platform, app feature, interface, or platform campaign?"
    ),
    "space exploration, photography": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show space or astronomy content, such as stars, planets, rockets, spacecraft, space stations, "
        "or educational content about space exploration?"
    ),
    "sportswear": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show sportswear-focused content, such as an athletic scene, sporty styling, "
        "or display or promotion of athletic clothing, shoes, or gear?"
    ),
    "chef/food": (
        f"{PROMPT_PREFIX} "
        "Question: Does this image clearly show food or cooking content, such as a dish, meal, cooking process, "
        "kitchen scene, restaurant setting, or food presentation?"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run personal-brand-specific yes/no VQA prompts with Qwen3-VL."
    )
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--csv_output_path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--mapping_json", default=None)
    parser.add_argument("--influencer_label_json", default=str(DEFAULT_INFLUENCER_LABEL_PATH))
    parser.add_argument("--labels_txt", default=str(DEFAULT_LABELS_TXT_PATH))

    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--flash_attn2", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--output_confidence", action="store_true")
    return parser.parse_args()


def resolve_dtype(dtype_name: str):
    lowered = dtype_name.lower()
    if lowered in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if lowered in {"float16", "fp16", "half"}:
        return torch.float16
    if lowered in {"float32", "fp32"}:
        return torch.float32
    return "auto"


def find_images(root: str) -> List[str]:
    paths = []
    for current_root, _, files in os.walk(root):
        for filename in files:
            if Path(filename).suffix.lower() in IMG_EXTS:
                paths.append(os.path.join(current_root, filename))
    paths.sort()
    return paths


def load_images(paths: List[str]) -> Tuple[List[Image.Image], List[str]]:
    images = []
    valid_paths = []
    for path in paths:
        try:
            images.append(Image.open(path).convert("RGB"))
            valid_paths.append(path)
        except Exception as exc:
            print(f"[SKIP] {path}: {exc}")
    return images, valid_paths


def compute_confidence(scores, sequences, input_ids, tokenizer):
    if scores is None:
        return [None] * sequences.size(0)

    gen_ids = sequences[:, input_ids.shape[1] :]
    eos_id = getattr(tokenizer, "eos_token_id", None)
    confidences = []

    for i in range(gen_ids.size(0)):
        tokens = gen_ids[i].tolist()
        length = len(tokens)
        if eos_id is not None and eos_id in tokens:
            length = tokens.index(eos_id)
        if length == 0:
            confidences.append(None)
            continue

        logprob_sum = 0.0
        count = 0
        steps = min(length, len(scores))
        for t in range(steps):
            logits = scores[t][i]
            logprob = torch.log_softmax(logits, dim=-1)
            logprob_sum += logprob[tokens[t]].item()
            count += 1

        if count == 0:
            confidences.append(None)
        else:
            confidences.append(float(math.exp(logprob_sum / count)))

    return confidences


def normalize_path(path_str: str) -> str:
    return str(Path(path_str).expanduser().resolve())


def infer_mapping_json(image_dir: Path, explicit_mapping: Optional[str]) -> Optional[Path]:
    if explicit_mapping:
        mapping_path = Path(explicit_mapping).expanduser().resolve()
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Mapping JSON not found: {mapping_path}")
        return mapping_path

    if not image_dir.name.startswith("sample_"):
        return None

    candidate = image_dir.parent / f"{image_dir.name}_mapping.json"
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(
        f"Input directory looks like a sample directory, but mapping JSON was not found: {candidate}"
    )


def load_mapping(mapping_path: Optional[Path]) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    if mapping_path is None:
        return {}, {}

    import json

    raw_mapping = json.load(open(mapping_path, "r", encoding="utf-8"))
    by_path: Dict[str, dict] = {}
    by_basename: Dict[str, dict] = {}

    for sample_path, entry in raw_mapping.items():
        if isinstance(entry, str):
            normalized_entry = {"original_path": entry}
        else:
            normalized_entry = dict(entry)

        normalized_key = normalize_path(sample_path)
        by_path[normalized_key] = normalized_entry

        basename = Path(sample_path).name
        if basename not in by_basename:
            by_basename[basename] = normalized_entry

    return by_path, by_basename


def extract_handle_from_original_path(original_path: str) -> Optional[str]:
    parts = Path(original_path).parts
    for idx, part in enumerate(parts):
        if part == "images" and idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def resolve_original_path_and_handle(
    image_path: str,
    mapping_by_path: Dict[str, dict],
    mapping_by_basename: Dict[str, dict],
) -> Tuple[str, str]:
    normalized_input = normalize_path(image_path)
    mapping_entry = mapping_by_path.get(normalized_input)
    if mapping_entry is None:
        mapping_entry = mapping_by_basename.get(Path(image_path).name)

    if mapping_entry is not None:
        original_path = mapping_entry.get("original_path", image_path)
        handle = extract_handle_from_original_path(original_path)
        if handle:
            return original_path, handle
        fallback_handle = mapping_entry.get("influencer")
        if fallback_handle:
            return original_path, fallback_handle
        raise ValueError(f"Could not determine handle from mapping entry for {image_path}")

    handle = extract_handle_from_original_path(image_path)
    if handle:
        return image_path, handle

    raise ValueError(
        f"Could not determine original path / handle for image: {image_path}. "
        "If this is a sample directory, pass --mapping_json explicitly."
    )


def load_owner_metadata(influencer_label_json: str) -> Dict[str, Dict[str, str]]:
    import json

    data = json.load(open(influencer_label_json, "r", encoding="utf-8"))
    result: Dict[str, Dict[str, str]] = {}
    for handle, info in data.items():
        brand = (info.get("personal_brand") or "").strip()
        known_name = (info.get("known_name") or "").strip()
        if brand:
            result[handle] = {
                "known_name": known_name,
                "personal_brand": brand,
            }
    return result


def load_personal_brand_map(influencer_label_json: str) -> Dict[str, str]:
    owner_metadata = load_owner_metadata(influencer_label_json)
    return {
        handle: info["personal_brand"]
        for handle, info in owner_metadata.items()
    }


def load_supported_labels(labels_txt: str) -> List[str]:
    lines = Path(labels_txt).read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def validate_prompt_coverage(supported_labels: List[str], personal_brand_map: Dict[str, str]) -> None:
    supported_set = set(supported_labels)
    prompt_set = set(PROMPTS_BY_LABEL.keys())

    if supported_set != prompt_set:
        missing_prompts = sorted(supported_set - prompt_set)
        extra_prompts = sorted(prompt_set - supported_set)
        raise ValueError(
            "Prompt coverage mismatch. "
            f"Missing prompt labels: {missing_prompts}. Extra prompt labels: {extra_prompts}."
        )

    label_values = set(personal_brand_map.values())
    unsupported = sorted(label_values - prompt_set)
    if unsupported:
        raise ValueError(
            f"Found personal_brand labels without prompts: {unsupported}"
        )


def prompt_for_personal_brand(personal_brand: str) -> str:
    try:
        return PROMPTS_BY_LABEL[personal_brand]
    except KeyError as exc:
        raise KeyError(f"Unsupported personal_brand label: {personal_brand}") from exc


def resolve_output_csv_path(csv_output_path: str) -> Path:
    output_path = Path(csv_output_path).expanduser()
    if output_path.suffix.lower() == ".csv":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path.resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    return (output_path / f"{output_path.name}.csv").resolve()


def normalize_yes_no(answer: str) -> str:
    stripped = (answer or "").strip()
    if not stripped:
        return "No answer generated."

    lowered = stripped.lower()
    if lowered.startswith("yes"):
        return "Yes"
    if lowered.startswith("no"):
        return "No"

    tokens = re.findall(r"[a-zA-Z]+", lowered)
    if tokens:
        if tokens[0] == "yes":
            return "Yes"
        if tokens[0] == "no":
            return "No"
    return stripped


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    mapping_path = infer_mapping_json(image_dir, args.mapping_json)
    mapping_by_path, mapping_by_basename = load_mapping(mapping_path)

    owner_metadata = load_owner_metadata(args.influencer_label_json)
    personal_brand_map = {
        handle: info["personal_brand"] for handle, info in owner_metadata.items()
    }
    supported_labels = load_supported_labels(args.labels_txt)
    validate_prompt_coverage(supported_labels, personal_brand_map)

    output_csv_path = resolve_output_csv_path(args.csv_output_path)

    image_paths = find_images(str(image_dir))
    if args.num_samples is not None:
        image_paths = image_paths[: args.num_samples]

    records = []
    label_counts: Dict[str, int] = {}
    for image_path in image_paths:
        original_path, handle = resolve_original_path_and_handle(
            image_path, mapping_by_path, mapping_by_basename
        )
        if handle not in owner_metadata:
            raise KeyError(
                f"Handle '{handle}' from image '{image_path}' was not found in influencer_label.json"
            )

        known_name = owner_metadata[handle]["known_name"]
        personal_brand = owner_metadata[handle]["personal_brand"]
        prompt = prompt_for_personal_brand(personal_brand)
        records.append(
            {
                "image_path": image_path,
                "original_path": original_path,
                "handle": handle,
                "known_name": known_name,
                "personal_brand": personal_brand,
                "prompt": prompt,
            }
        )
        label_counts[personal_brand] = label_counts.get(personal_brand, 0) + 1

    print(f"Found {len(records)} images")
    if mapping_path is not None:
        print(f"Using mapping JSON: {mapping_path}")
    print("Personal brand distribution in this run:")
    for label in sorted(label_counts):
        print(f"  {label}: {label_counts[label]}")

    dtype = resolve_dtype(args.dtype)
    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    attn_impl = "flash_attention_2" if args.flash_attn2 else "sdpa"
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=dtype,
        attn_implementation=attn_impl,
        device_map=args.device_map,
    )
    model.eval()
    device = model.device
    print(
        f"Loaded model on {device}; dtype={model.dtype}; flash_attn2={args.flash_attn2}"
    )

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "answer", "known_name", "personal_brand"],
        )
        writer.writeheader()

        for i in tqdm(range(0, len(records), args.batch_size), desc="PersonalBrand"):
            batch_records = records[i : i + args.batch_size]
            batch_paths = [record["image_path"] for record in batch_records]
            batch_images, valid_paths = load_images(batch_paths)

            if not batch_images:
                continue

            valid_record_map = {record["image_path"]: record for record in batch_records}
            valid_records = [valid_record_map[path] for path in valid_paths]

            messages = []
            for img, record in zip(batch_images, valid_records):
                image_content = {"type": "image", "image": img}
                if args.min_pixels is not None:
                    image_content["min_pixels"] = args.min_pixels
                if args.max_pixels is not None:
                    image_content["max_pixels"] = args.max_pixels

                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                image_content,
                                {"type": "text", "text": record["prompt"]},
                            ],
                        }
                    ]
                )

            texts = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )

            if videos is not None:
                videos, video_metadatas = zip(*videos)
                videos, video_metadatas = list(videos), list(video_metadatas)
            else:
                video_metadatas = None

            inputs = processor(
                text=texts,
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                do_resize=False,
                **video_kwargs,
            ).to(device)

            generation_kwargs = dict(gen_kwargs)
            if args.output_confidence:
                generation_kwargs.update(
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            with torch.no_grad():
                outputs = model.generate(**inputs, **generation_kwargs)

            if args.output_confidence:
                sequences = outputs.sequences
                scores = outputs.scores
            else:
                sequences = outputs
                scores = None

            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, sequences)]
            answers = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            confidences = (
                compute_confidence(
                    scores, sequences, inputs.input_ids, processor.tokenizer
                )
                if args.output_confidence
                else [None] * len(answers)
            )

            for record, answer, confidence in zip(valid_records, answers, confidences):
                normalized_answer = normalize_yes_no(answer.replace("\n", " ").strip())
                if confidence is None:
                    print(
                        f"{record['image_path']} | {record['personal_brand']} -> {normalized_answer}"
                    )
                else:
                    print(
                        f"{record['image_path']} | {record['personal_brand']} -> "
                        f"{normalized_answer} (conf={confidence:.4f})"
                    )
                writer.writerow(
                    {
                        "filename": record["image_path"],
                        "answer": normalized_answer,
                        "known_name": record["known_name"],
                        "personal_brand": record["personal_brand"],
                    }
                )

    print(f"Saved rows -> {output_csv_path}")


if __name__ == "__main__":
    main()
