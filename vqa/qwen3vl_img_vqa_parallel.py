#!/usr/bin/env python3
import argparse
import os
import csv
import random
from typing import List

from PIL import Image
from tqdm import tqdm
from pillow_heif import register_heif_opener

import torch
import torch.distributed as dist

from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

register_heif_opener()

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif")


def setup_distributed():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def find_files(root: str, exts: List[str]) -> List[str]:
    paths = []
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(tuple(exts)):
                paths.append(os.path.join(r, f))
    paths.sort()
    return paths


def partition_indices(total_size: int, world_size: int, rank: int) -> List[int]:
    per_rank = total_size // world_size
    rem = total_size % world_size
    if rank < rem:
        start = rank * (per_rank + 1)
        end = start + (per_rank + 1)
    else:
        start = rank * per_rank + rem
        end = start + per_rank
    return list(range(start, end))


def load_images(paths: List[str]) -> (List[Image.Image], List[bool]):
    imgs, mask = [], []
    for p in paths:
        try:
            imgs.append(Image.open(p).convert("RGB"))
            mask.append(True)
        except Exception as e:
            print(f"[WARN] Failed to open {p}: {e}")
            imgs.append(Image.new("RGB", (224, 224), (255, 255, 255)))
            mask.append(False)
    return imgs, mask


def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL Image VQA (multi-GPU via torchrun)")
    parser.add_argument("--image_dir", required=True, help="Root dir of images (recursive).")
    parser.add_argument("--output_dir", default="qwen3vl_img_vqa_ddp", help="Output root dir.")
    parser.add_argument("--questions", nargs="+", default=["Describe the image."], help="Questions to ask.")
    parser.add_argument("--num_samples", type=int, default=None, help="Global cap on images (after sort/shuffle).")
    parser.add_argument("--batch_size", type=int, default=64, help="Per-GPU batch size.")
    parser.add_argument("--num_workers", type=int, default=4, help="(Reserved) Not used; PIL loading inline.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle list before applying --num_samples.")
    parser.add_argument("--seed", type=int, default=2024, help="Shuffle seed.")

    parser.add_argument("--model_name", default="Qwen/Qwen3-VL-8B-Instruct", help="HF model id or local path.")
    parser.add_argument("--dtype", default="auto", help="auto|bfloat16|float16|float32")
    parser.add_argument("--flash_attn2", action="store_true", help="Enable Flash-Attention 2 (if installed).")

    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    parser.add_argument("--min_pixels", type=int, default=None)
    parser.add_argument("--max_pixels", type=int, default=None)

    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if args.shuffle:
        random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    image_paths = find_files(args.image_dir, list(IMG_EXTS))
    if args.shuffle:
        random.shuffle(image_paths)
    if args.num_samples is not None:
        image_paths = image_paths[: args.num_samples]

    total_images = len(image_paths)
    if rank == 0:
        print(f"[DDP] world_size={world_size} | total_images={total_images} | per-gpu batch={args.batch_size}")

    my_indices = partition_indices(total_images, world_size, rank)
    my_paths = [image_paths[i] for i in my_indices]
    print(f"[Rank {rank}] assigned {len(my_paths)} images")

    dtype: object = "auto"
    if args.dtype.lower() in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    elif args.dtype.lower() in ("float16", "fp16", "half"):
        dtype = torch.float16
    elif args.dtype.lower() in ("float32", "fp32"):
        dtype = torch.float32

    attn_impl = "flash_attention_2" if args.flash_attn2 else "sdpa"

    processor = AutoProcessor.from_pretrained(args.model_name)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=dtype,
        attn_implementation=attn_impl,
        device_map=None,
    )
    model = model.to(device)
    model.eval()
    if rank == 0:
        print(f"[Model] loaded {args.model_name} on {device} | dtype={model.dtype} | attn={attn_impl}")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
    )

    for q_idx, question in enumerate(args.questions, start=1):
        q_dir = os.path.join(args.output_dir, f"question_{q_idx}")
        os.makedirs(q_dir, exist_ok=True)

        rank_csv = os.path.join(q_dir, f"rank_{rank}.csv")
        rank_log = os.path.join(q_dir, f"rank_{rank}.empty_answers.log")

        with open(rank_csv, "w", newline="", encoding="utf-8") as fcsv, \
            open(rank_log, "w", encoding="utf-8") as flog:

            writer = csv.DictWriter(fcsv, fieldnames=["filename", "answer"])
            writer.writeheader()
            flog.write("Empty Answers Log\n=================\n")

            if rank == 0:
                iterator = tqdm(range(0, len(my_paths), args.batch_size), desc=f"Q{q_idx}")
            else:
                iterator = range(0, len(my_paths), args.batch_size)

            for i in iterator:
                batch_paths = my_paths[i : i + args.batch_size]
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
                    outputs = model.generate(**inputs, **gen_kwargs)

                trimmed = [out[len(inp) :] for inp, out in zip(inputs.input_ids, outputs)]
                answers = processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                for p, ans, ok in zip(batch_paths, answers, valid_mask):
                    ans = (ans or "").strip().replace("\n", " ")
                    if ans.startswith("addCriterion"):
                        ans = ans.replace("addCriterion", "", 1).strip()
                    if not ans:
                        flog.write(f"Empty answer for image: {p}\n")
                        ans = "No answer generated."
                    if not ok:
                        ans = f"[LOAD_ERROR] {ans}"
                    writer.writerow({"filename": p, "answer": ans})

        dist.barrier()
        if rank == 0:
            final_csv = os.path.join(q_dir, "results.csv")
            final_log = os.path.join(q_dir, "empty_answers.log")
            with open(final_csv, "w", newline="", encoding="utf-8") as fout:
                writer = csv.DictWriter(fout, fieldnames=["filename", "answer"])
                writer.writeheader()
                for r in range(world_size):
                    part = os.path.join(q_dir, f"rank_{r}.csv")
                    if not os.path.exists(part):
                        continue
                    with open(part, "r", encoding="utf-8") as fin:
                        next(fin, None)
                        for line in fin:
                            fout.write(line)

            with open(final_log, "w", encoding="utf-8") as lf:
                lf.write("Empty Answers Log\n=================\n")
                for r in range(world_size):
                    part = os.path.join(q_dir, f"rank_{r}.empty_answers.log")
                    if not os.path.exists(part):
                        continue
                    with open(part, "r", encoding="utf-8") as pf:
                        lines = pf.readlines()
                        lf.writelines(lines[2:] if len(lines) > 2 else [])

            for r in range(world_size):
                for nm in (f"rank_{r}.csv", f"rank_{r}.empty_answers.log"):
                    pth = os.path.join(q_dir, nm)
                    if os.path.exists(pth):
                        try:
                            os.remove(pth)
                        except Exception:
                            pass

            print(f"[Q{q_idx}] merged -> {final_csv}")

        dist.barrier()

    cleanup_distributed()
    if rank == 0:
        print("All questions processed.")


if __name__ == "__main__":
    main()
