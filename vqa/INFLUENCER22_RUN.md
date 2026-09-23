# Influencer_22 VQA run on the cluster

Runs the 11 active v3 prompts from `vqa/prompt_test.ipynb` over the full
influencer_22 image set with the multi-GPU runner already on `main`
(`vqa/qwen3vl_img_vqa_parallel_v5.py`).

## What is needed

| Item | Where | Notes |
|---|---|---|
| `vqa/qwen3vl_img_vqa_parallel_v5.py` | already on `main` | unchanged |
| `vqa/bash_qwen3vl_img_vqa_influencer22.sh` | `local` branch | new launcher with the 11 prompts |
| influencer_22 images | cluster | point `IMAGE_DIR` at the `images/` folder |

No label files are needed. All 11 prompts use only the image.
`influencer_label.json`, `labels.txt` and the `sample_*_mapping.json` files
are used only by `personal_brand.py` and `main_job.py`, which are not part of
this run. The influencer handle and post id can be read from each output row's
`filename` (`.../images/<influencer>/<post>/<file>`).

## Steps

1. Get the launcher onto the cluster checkout of `main`:

   ```bash
   cd /proj/berzelius-2024-409/users/x_liumi/Qwen3-VL
   git fetch origin local
   git checkout origin/local -- vqa/bash_qwen3vl_img_vqa_influencer22.sh vqa/INFLUENCER22_RUN.md
   ```

2. Check the dataset copy. Expect about 20,138 image files (20,137 usable, one
   zero-byte file under `ellendegeneres`):

   ```bash
   find /path/to/influencer_22/images -type f \
     \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | wc -l
   ```

3. Edit the top of `vqa/bash_qwen3vl_img_vqa_influencer22.sh`:
   - `IMAGE_DIR`: the cluster path to `influencer_22/images`
   - `OUTPUT_DIR`: a new, empty folder (existing `results.csv` files are overwritten)
   - `NUM_GPUS`: the number of GPUs on the allocated node
   - `SCRIPT`: the absolute path to `vqa/qwen3vl_img_vqa_parallel_v5.py` if the repo lives elsewhere

4. Run it on a GPU node, the same way as the old `v5_*.sh` scripts:

   ```bash
   conda activate qwen3vl   # or the environment used for the v5 runs
   bash vqa/bash_qwen3vl_img_vqa_influencer22.sh 2>&1 | tee influencer22_v3.log
   ```

   For a quick test first, set `NUM_SAMPLES=80` and use a temporary `OUTPUT_DIR`.

## Output

```
OUTPUT_DIR/
  question_1/results.csv      # filename,answer
  question_1/empty_answers.log
  ...
  question_11/results.csv
```

Question numbers follow the notebook order:

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

Differences from the notebook output:
- Each `answer` ends with ` (conf=0.xxxx)`. The reformat cell in
  `preprocessing_inf.ipynb` still parses the JSON, because it extracts the
  `{...}` part.
- An image that fails to load gets a `[LOAD_ERROR]` prefix instead of being skipped.
- The files are `question_N/results.csv` instead of `<folder>_qN.csv`. To use the
  reformat cell, flatten them first:

  ```bash
  mkdir -p OUTPUT_DIR_flat
  for d in OUTPUT_DIR/question_*; do
    n=${d##*_}; cp "$d/results.csv" "OUTPUT_DIR_flat/influencer22_q${n}.csv"
  done
  ```

## Settings kept from the notebook

`MAX_NEW_TOKENS=500`, `TEMPERATURE=0.6`, `TOP_P=0.9`, `TOP_K=50`,
`REPETITION_PENALTY=1.05`, `bfloat16`, `BATCH_SIZE=4`, `MIN_PIXELS`/`MAX_PIXELS` unset.
The old cluster limit (`MAX_PIXELS=1280*32*32`) is commented out in the
launcher. Turn it on only if GPU memory runs out. It slightly downsizes
full-resolution Instagram images, so results will differ a little from the pilot.

## Troubleshooting

- NCCL timeout at a question barrier (one rank is much slower than the others):
  set `SCRIPT` to `qwen3vl_img_vqa_parallel_v5_copy.py`, which is also on `main`
  and uses a 30-minute timeout.
- CUDA out of memory: lower `BATCH_SIZE` to 2, or enable `MAX_PIXELS`.
- Port already in use: change `RDZV_ENDPOINT` to another port, such as `localhost:29501`.
