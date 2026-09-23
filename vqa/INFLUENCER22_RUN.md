# Influencer_22 VQA run on the cluster

Runs the 12-question v3 set over the full influencer_22 image set:

- Part A (Q1-Q11): the 11 active prompts in `vqa/prompt_test.ipynb`. They use only
  the image and run with the multi-GPU runner already on `main`
  (`vqa/qwen3vl_img_vqa_parallel_v5.py`).
- Part B (Q12): the main-job yes/no prompt in `vqa/main_job.py`. It builds each
  prompt from the image owner's `known_name` and `personal_brand`.

## What is needed

| Item | Source | Used by |
|---|---|---|
| `vqa/qwen3vl_img_vqa_parallel_v5.py` | already on `main` | Part A |
| `vqa/bash_qwen3vl_img_vqa_influencer22.sh` | `local` branch | Part A launcher |
| `vqa/main_job.py`, `vqa/personal_brand.py` | `local` branch | Part B (`main_job.py` imports helpers from `personal_brand.py`) |
| `vqa/bash_qwen3vl_img_vqa_influencer22_main_job.sh` | `local` branch | Part B launcher |
| `data_influencer22/influencer_label.json` | copy by hand (not in git) | Part B: handle -> `known_name`, `personal_brand` |
| influencer_22 images | cluster | both parts; point `IMAGE_DIR` at the `images/` folder |

`labels.txt` and the `sample_*_mapping.json` files are not needed.

Use `influencer_label.json` from `data_influencer22/`. It is identical to the
copy in `data_influencer22_v4_account_level/` and covers all 122 handles. The
older `data_influencer22_v3/` copy has only 102 handles, so `main_job.py` would
stop with a `KeyError`.

Part B reads the owner handle from the path `.../images/<handle>/<post>/<file>`,
so `IMAGE_DIR` must keep that layout, and no other folder in the path may be
named `images`. A local dry run over all 20,138 images resolved every image to
a handle and a prompt.

## Steps

1. Get the code onto the cluster checkout of `main`:

   ```bash
   cd /proj/berzelius-2024-409/users/x_liumi/Qwen3-VL
   git fetch origin local
   git checkout origin/local -- \
     vqa/bash_qwen3vl_img_vqa_influencer22.sh \
     vqa/bash_qwen3vl_img_vqa_influencer22_main_job.sh \
     vqa/main_job.py vqa/personal_brand.py \
     vqa/INFLUENCER22_RUN.md
   ```

2. Copy the label file from the local machine:

   ```bash
   # run on the local machine
   ssh <cluster> mkdir -p /proj/berzelius-2024-409/users/x_liumi/Qwen3-VL/data_influencer22
   scp /home/labad/minxing/code/Qwen3-VL/data_influencer22/influencer_label.json \
     <cluster>:/proj/berzelius-2024-409/users/x_liumi/Qwen3-VL/data_influencer22/
   ```

   The Part B launcher expects it at `$REPO_DIR/data_influencer22/influencer_label.json`.
   If you put it somewhere else, change `INFLUENCER_LABEL_JSON`.

3. Check the dataset copy. Expect about 20,138 image files (20,137 usable, one
   zero-byte file under `ellendegeneres`):

   ```bash
   find /path/to/influencer_22/images -type f \
     \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l
   ```

4. Part A: edit the top of `vqa/bash_qwen3vl_img_vqa_influencer22.sh`:
   - `IMAGE_DIR`: the cluster path to `influencer_22/images`
   - `OUTPUT_DIR`: a new, empty folder (existing `results.csv` files are overwritten)
   - `NUM_GPUS`: the number of GPUs on the allocated node
   - `SCRIPT`: the absolute path to `vqa/qwen3vl_img_vqa_parallel_v5.py` if the repo lives elsewhere

   Then run it on a GPU node, the same way as the old `v5_*.sh` scripts:

   ```bash
   conda activate qwen3vl   # or the environment used for the v5 runs
   bash vqa/bash_qwen3vl_img_vqa_influencer22.sh 2>&1 | tee influencer22_v3.log
   ```

   For a quick test first, set `NUM_SAMPLES=80` and use a temporary `OUTPUT_DIR`.

5. Part B: edit the top of `vqa/bash_qwen3vl_img_vqa_influencer22_main_job.sh`
   (`REPO_DIR`, `IMAGE_DIR`, `OUTPUT_DIR`, `GPU`). Use the same `OUTPUT_DIR` as
   Part A to keep all 12 answers together. Then run:

   ```bash
   bash vqa/bash_qwen3vl_img_vqa_influencer22_main_job.sh
   ```

   `main_job.py` is single-process and uses one GPU. Run it after Part A, or
   give it a GPU that Part A is not using (for example `NUM_GPUS=7` for Part A
   and `GPU=7` here). The log is written to `OUTPUT_DIR/main_job.log`.

## Output

```
OUTPUT_DIR/
  question_1/results.csv      # filename,answer
  question_1/empty_answers.log
  ...
  question_11/results.csv
  <OUTPUT_DIR name>_q12.csv   # filename,answer,known_name,personal_brand
  main_job.log
```

| N | Prompt |
|---|---|
| 1 | Communicative intent |
| 2 | Facial beautification (1-3 scale) |
| 3 | Facial beautification (1-5 scale) |
| 4 | Global stylization or editing |
| 5 | Sexual suggestiveness |
| 6 | Image-level emotion |
| 7 | Topic |
| 8 | Person |
| 9 | Attractiveness |
| 10 | Image type |
| 11 | Image-level promotion |
| 12 | Main job (Yes/No) |

Differences in Part A from the notebook output:
- Each `answer` ends with ` (conf=0.xxxx)`. The reformat cell in
  `preprocessing_inf.ipynb` still parses the JSON, because it extracts the
  `{...}` part.
- An image that fails to load gets a `[LOAD_ERROR]` prefix instead of being
  skipped. `main_job.py` skips such images instead.
- The files are `question_N/results.csv` instead of `<folder>_qN.csv`. To use the
  reformat cell, flatten them first. The Q12 file is already in that format:

  ```bash
  mkdir -p OUTPUT_DIR_flat
  for d in OUTPUT_DIR/question_*; do
    n=${d##*_}; cp "$d/results.csv" "OUTPUT_DIR_flat/influencer22_q${n}.csv"
  done
  cp OUTPUT_DIR/*_q12.csv OUTPUT_DIR_flat/influencer22_q12.csv
  ```

## Settings kept from the notebook

Part A: `MAX_NEW_TOKENS=500`, `TEMPERATURE=0.6`, `TOP_P=0.9`, `TOP_K=50`,
`REPETITION_PENALTY=1.05`, `bfloat16`, `BATCH_SIZE=4`, `MIN_PIXELS`/`MAX_PIXELS` unset.
Part B: the `main_job.py` defaults, as in the local pilot: `max_new_tokens=32`,
with the model's own sampling settings from `generation_config.json`
(`temperature=0.7`, `top_p=0.8`, `top_k=20`).

The old cluster limit (`MAX_PIXELS=1280*32*32`) is commented out in the Part A
launcher. Turn it on only if GPU memory runs out. It slightly downsizes
full-resolution Instagram images, so results will differ a little from the pilot.

## Troubleshooting

- NCCL timeout at a question barrier in Part A (one rank is much slower than the
  others): set `SCRIPT` to `qwen3vl_img_vqa_parallel_v5_copy.py`, which is also
  on `main` and uses a 30-minute timeout.
- CUDA out of memory: lower `BATCH_SIZE` to 2, or enable `MAX_PIXELS`.
- Port already in use: change `RDZV_ENDPOINT` to another port, such as `localhost:29501`.
- `KeyError: Handle '...' was not found in influencer_label.json`: the wrong
  label file was copied. Use the one from `data_influencer22/`.
