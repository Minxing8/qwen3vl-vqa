#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_vl_utils import process_vision_info

try:
    from personal_brand import (
        REPO_ROOT,
        find_images,
        infer_mapping_json,
        load_images,
        load_mapping,
        normalize_yes_no,
        resolve_dtype,
        resolve_original_path_and_handle,
    )
except ImportError:
    from vqa.personal_brand import (
        REPO_ROOT,
        find_images,
        infer_mapping_json,
        load_images,
        load_mapping,
        normalize_yes_no,
        resolve_dtype,
        resolve_original_path_and_handle,
    )


DEFAULT_IMAGE_DIR = REPO_ROOT / "data_influencer22" / "sample_50"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "data_influencer22" / "output" / "test_50"
DEFAULT_INFLUENCER_LABEL_PATH = REPO_ROOT / "data_influencer22" / "influencer_label.json"
DEFAULT_QUESTION_INDEX = 12

BRAND_BASED_PROMPT_NAMES = {"433", "9GAG"}


def resolve_output_csv_path(csv_output_path: str, question_index: int) -> Path:
    """
    Match the q-suffix naming used by prompt_test.ipynb so this script's output
    sits alongside the other answer files as `<folder>_q<N>.csv`.

    - If `csv_output_path` already ends in `.csv`, it is used as-is.
    - Otherwise it is treated as a folder; the file is written as
      `<folder>/<folder.name>_q{question_index}.csv`.
    """
    output_path = Path(csv_output_path).expanduser()
    if output_path.suffix.lower() == ".csv":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path.resolve()

    output_path.mkdir(parents=True, exist_ok=True)
    return (output_path / f"{output_path.name}_q{question_index}.csv").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simplified owner/main-job VQA prompt with Qwen3-VL."
    )
    parser.add_argument("--image_dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument(
        "--csv_output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Folder to write the answer CSV into (the file will be named "
            "<folder>_q<question_index>.csv to match prompt_test.ipynb), "
            "or an explicit .csv path."
        ),
    )
    parser.add_argument(
        "--question_index",
        type=int,
        default=DEFAULT_QUESTION_INDEX,
        help="Question number suffix used in the output filename (default: 12).",
    )
    parser.add_argument("--mapping_json", default=None)
    parser.add_argument("--influencer_label_json", default=str(DEFAULT_INFLUENCER_LABEL_PATH))

    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--flash_attn2", action="store_true")
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)
    return parser.parse_args()


def load_owner_metadata(influencer_label_json: str) -> Dict[str, Dict[str, str]]:
    data = json.load(open(influencer_label_json, "r", encoding="utf-8"))
    result: Dict[str, Dict[str, str]] = {}

    for handle, info in data.items():
        known_name = (info.get("known_name") or "").strip()
        personal_brand = (info.get("personal_brand") or "").strip()

        if not known_name:
            raise ValueError(f"Missing known_name for handle: {handle}")
        if not personal_brand:
            raise ValueError(f"Missing personal_brand for handle: {handle}")

        result[handle] = {
            "known_name": known_name,
            "personal_brand": personal_brand,
        }

    return result


def build_prompt(known_name: str, personal_brand: str) -> str:
    if known_name in BRAND_BASED_PROMPT_NAMES:
        return (
            "Answer exactly one word: Yes or No. "
            "Do not explain. "
            f"Is this image related to {personal_brand}?"
        )

    return (
        "Answer exactly one word: Yes or No. "
        "Do not explain. "
        f"Is this image related to the main business or primary professional activity of {known_name}?"
    )


def main() -> None:
    args = parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {image_dir}")

    mapping_path = infer_mapping_json(image_dir, args.mapping_json)
    mapping_by_path, mapping_by_basename = load_mapping(mapping_path)
    owner_metadata = load_owner_metadata(args.influencer_label_json)
    output_csv_path = resolve_output_csv_path(args.csv_output_path, args.question_index)

    image_paths = find_images(str(image_dir))
    if args.num_samples is not None:
        image_paths = image_paths[: args.num_samples]
    if not image_paths:
        raise ValueError(f"No images found under: {image_dir}")

    records = []
    for image_path in image_paths:
        original_path, handle = resolve_original_path_and_handle(
            image_path, mapping_by_path, mapping_by_basename
        )
        info = owner_metadata.get(handle)
        if info is None:
            raise KeyError(
                f"Handle '{handle}' from image '{image_path}' was not found in influencer_label.json"
            )

        known_name = info["known_name"]
        personal_brand = info["personal_brand"]
        prompt = build_prompt(known_name, personal_brand)

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

    print(f"Found {len(records)} images")
    if mapping_path is not None:
        print(f"Using mapping JSON: {mapping_path}")

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

    with open(output_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["filename", "answer", "known_name", "personal_brand"],
        )
        writer.writeheader()

        for i in tqdm(range(0, len(records), args.batch_size), desc="MainJob"):
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

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                )

            trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, outputs)]
            answers = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            for record, answer in zip(valid_records, answers):
                normalized_answer = normalize_yes_no(answer.replace("\n", " ").strip())
                print(
                    f"{record['image_path']} | {record['handle']} | "
                    f"{record['known_name']} -> {normalized_answer}"
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
