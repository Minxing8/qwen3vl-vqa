#!/usr/bin/env python3
import argparse
import os
import csv
from typing import List

from tqdm import tqdm
from PIL import Image
import torch
from pillow_heif import register_heif_opener

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

register_heif_opener()


def find_files(root: str, exts: List[str]) -> List[str]:
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(tuple(exts)):
                paths.append(os.path.join(r, f))
    paths.sort()
    return paths


def load_images(paths: List[str]) -> (List[Image.Image], List[bool]):
    imgs, mask = [], []
    for pth in paths:
        try:
            imgs.append(Image.open(pth).convert("RGB"))
            mask.append(True)
        except Exception as e:
            print(f"[WARN] Failed to open {pth}: {e}")
            imgs.append(Image.new("RGB", (224, 224), (255, 255, 255)))
            mask.append(False)
    return imgs, mask


def main():
    p = argparse.ArgumentParser(description="Qwen3-VL Image VQA (batched)")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--output_dir", default="qwen3vl_img_vqa")
    p.add_argument("--questions", nargs="+", default=["Describe the image."])
    p.add_argument("--num_samples", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--repetition_penalty", type=float, default=1.05)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--flash_attn2", action="store_true")
    p.add_argument("--device_map", default="auto")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dtype: object = "auto"
    if args.dtype.lower() in ["bfloat16", "bf16"]:
        dtype = torch.bfloat16
    elif args.dtype.lower() in ["float16", "fp16", "half"]:
        dtype = torch.float16
    elif args.dtype.lower() in ["float32", "fp32"]:
        dtype = torch.float32

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
    print(f"Loaded model on {device}; dtype={model.dtype}; flash_attn2={args.flash_attn2}")

    image_paths = find_files(args.image_dir, [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"])
    if args.num_samples is not None:
        image_paths = image_paths[: args.num_samples]
    print(f"Total images: {len(image_paths)}")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    for q_idx, question in enumerate(args.questions, start=1):
        print(f"\n[Q{q_idx}] {question}")
        q_out_dir = os.path.join(args.output_dir, f"question_{q_idx}")
        os.makedirs(q_out_dir, exist_ok=True)
        csv_path = os.path.join(q_out_dir, "results.csv")
        empty_log = os.path.join(q_out_dir, "empty_answers.log")

        with open(csv_path, "w", newline="", encoding="utf-8") as fcsv, \
            open(empty_log, "w", encoding="utf-8") as flog:

            writer = csv.DictWriter(fcsv, fieldnames=["filename", "answer"])
            writer.writeheader()
            flog.write("Empty Answers Log\n=================\n")

            for i in tqdm(range(0, len(image_paths), args.batch_size), desc=f"Q{q_idx}"):
                batch_paths = image_paths[i : i + args.batch_size]
                batch_imgs, valid_mask = load_images(batch_paths)

                messages = []
                for img in batch_imgs:
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
                                    {"type": "text", "text": question},
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
                    generated = model.generate(**inputs, **gen_kwargs)

                trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, generated)]
                answers = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for pth, ans, is_valid in zip(batch_paths, answers, valid_mask):
                    ans = (ans or "").strip().replace("\n", " ")
                    if ans.startswith("addCriterion"):
                        ans = ans.replace("addCriterion", "", 1).strip()
                    if not ans:
                        flog.write(f"Empty answer for image: {pth}\n")
                        ans = "No answer generated."
                    if not is_valid:
                        ans = f"[LOAD_ERROR] {ans}"
                    writer.writerow({"filename": pth, "answer": ans})

        print(f"[Saved] {csv_path}")

    print("All questions processed.")


if __name__ == "__main__":
    main()
