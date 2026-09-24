#!/usr/bin/env python3
"""
Turn the raw influencer_22 VQA output into annotation-ready CSVs.

Input  (written by vqa/bash_qwen3vl_img_vqa_parallel_influencer.sh):
    <raw>/question_1/results.csv ... <raw>/question_11/results.csv
    <raw>/influencer_q12.csv
Output:
    <dest>/influencer_q01.csv ... <dest>/influencer_q12.csv

Two steps:

1. Repair. A handful of answers are not valid JSON because the model mangled
   text it read off the image: an invalid `\\'` escape, an unescaped `"` inside
   a string, a missing comma between two array strings, or a string value that
   closes early and then continues unquoted. Each broken answer is repaired and
   re-checked; the raw files are never modified.

2. Reformat. Every JSON key becomes its own column (nested keys flattened with
   dot notation, lists of primitives joined with " | "), plus a `manual_TF`
   column and a `manual_<key>` column for each key that is actually labellable.
   This mirrors the reformat cell of vqa/preprocessing_inf.ipynb on the `local`
   branch; keep the two in sync if either changes.
"""

import argparse
import csv
import glob
import json
import os
import re

import pandas as pd

csv.field_size_limit(10**9)

DEFAULT_RAW = "/proj/berzelius-2024-409/users/x_liumi/Qwen3-VL/output/Qwen3-VL/output/influencer"
DEFAULT_DEST = "/proj/berzelius-2024-409/users/x_liumi/Qwen3-VL/output/Qwen3-VL/output/influencer_reformed"

# Ordinal level sets (lower-cased). A column whose unique values are a subset
# of one of these is classified as 'ordinal' and gets a manual column.
ORDINAL_SETS = [
    {"low", "medium", "high"},
    {"low", "medium", "high", "very high"},
    {"none", "low", "medium", "high"},
    {"none", "low", "medium", "high", "very high"},
    {"very low", "low", "medium", "high", "very high"},
    {"mild", "moderate", "severe"},
    {"never", "sometimes", "often", "always"},
    {"strongly disagree", "disagree", "neutral", "agree", "strongly agree"},
    {"not present", "slightly present", "moderately present", "strongly present"},
    {"absent", "slight", "moderate", "strong"},
    {"positive", "neutral_or_mixed", "negative"},
]

# Leaf field names that must NEVER get a manual column - either free text
# (nothing for the annotator to tick) or the model's own confidence score.
NO_MANUAL_LEAF_NAMES = {
    "confidence",
    "reasoning",
    "justification",
    "attractiveness_justification",
    "visual_evidence",
    "evidence",
    "name",
    "outfit",
    "activity_or_pose",
    "facial_expression",
    "interaction_with_others",
    "promoted_entity",
    "other_topic",
    "other_label",
    "notes",
    "description",
}

FREE_TEXT_SUFFIXES = ("_reasoning", "_justification", "_evidence", "_notes")

# Categorical columns with more unique values than this are treated as free
# text (no manual column). The topic set of question 7 has 16 members, so this
# has to sit above 16.
MAX_CATEGORICAL_UNIQUE = 20


# -- repair --------------------------------------------------------------------

CONF_RE = re.compile(r"\s*\(conf=[\d.]+\)\s*$")

# `"key": "some value" trailing words,` -> `"key": "some value trailing words",`
UNQUOTED_TAIL_RE = re.compile(r'(:\s*)"([^"\n]*)"[ \t]+([^",:{}\[\]\n]+?)\s*(,\s*")')


def parses(text):
    """True if the `{...}` extraction below would yield a dict."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return False
    try:
        return isinstance(json.loads(m.group()), dict)
    except Exception:
        return False


def fold_unquoted_tail(text):
    return UNQUOTED_TAIL_RE.sub(
        lambda m: '%s"%s %s"%s' % (m.group(1), m.group(2), m.group(3), m.group(4)), text
    )


def repair(text):
    """Walk the answer and close, escape or separate what the model left broken."""
    out = []
    i, n = 0, len(text)
    in_str = False
    esc = False
    while i < n:
        c = text[i]
        if not in_str:
            if c == '"':
                in_str = True
            out.append(c)
            i += 1
            continue
        if esc:
            out.append(c)
            esc = False
            i += 1
            continue
        if c == "\\":
            if i + 1 < n and text[i + 1] not in '"\\/bfnrtu':
                i += 1          # drop the backslash of an invalid escape
                continue
            out.append(c)
            esc = True
            i += 1
            continue
        if c == '"':
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j >= n or text[j] in ",:}]":
                in_str = False          # a legitimate closing quote
                out.append(c)
            elif text[j] == '"':
                in_str = False          # two strings with no comma between them
                out.append('",')
            else:
                out.append('\\"')       # a stray quote inside the value
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def stage_repaired(raw_dir, dest_dir):
    """Copy the 12 raw files into dest_dir, repairing broken answers on the way."""
    os.makedirs(dest_dir, exist_ok=True)
    report = []

    for q in range(1, 12):
        src = os.path.join(raw_dir, "question_%d" % q, "results.csv")
        dst = os.path.join(dest_dir, "influencer_q%02d.csv" % q)
        rows = list(csv.DictReader(open(src, newline="", encoding="utf-8")))

        broken = repaired = null_only = 0
        for row in rows:
            body = CONF_RE.sub("", row["answer"])
            if parses(body):
                continue
            if body.strip() == "null":
                null_only += 1      # the prompt allows a bare null answer
                continue
            broken += 1
            for candidate in (repair(body), repair(fold_unquoted_tail(body))):
                if parses(candidate):
                    repaired += 1
                    suffix = CONF_RE.search(row["answer"])
                    row["answer"] = candidate + (suffix.group() if suffix else "")
                    break

        with open(dst, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "answer"])
            writer.writeheader()
            writer.writerows(rows)

        report.append((os.path.basename(dst), len(rows), broken, repaired, null_only))

    q12_src = os.path.join(raw_dir, "influencer_q12.csv")
    q12_dst = os.path.join(dest_dir, "influencer_q12.csv")
    rows12 = list(csv.DictReader(open(q12_src, newline="", encoding="utf-8")))
    with open(q12_dst, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows12[0].keys())
        writer.writeheader()
        writer.writerows(rows12)
    report.append((os.path.basename(q12_dst), len(rows12), 0, 0, 0))

    print("%-22s %8s %8s %9s %10s" % ("file", "rows", "broken", "repaired", "bare_null"))
    for name, rows_n, broken, rep, nulls in report:
        print("%-22s %8d %8d %9d %10d" % (name, rows_n, broken, rep, nulls))
    print()


# -- JSON extraction -----------------------------------------------------------

def extract_json(text):
    """Return a parsed dict/list from a JSON-ish string, or None on failure."""
    if not isinstance(text, str):
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    cleaned = cleaned.replace("'", '"')
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass

    return None


# -- flattening ----------------------------------------------------------------

def flatten_dict(d, prefix=""):
    """Recursively flatten a dict into {dotted.key: scalar} pairs."""
    items = {}
    if not isinstance(d, dict):
        return items
    for k, v in d.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, full_key))
        elif isinstance(v, list):
            if v and isinstance(v[0], dict):
                for i, elem in enumerate(v):
                    if isinstance(elem, dict):
                        items.update(flatten_dict(elem, f"{full_key}.{i}"))
                    else:
                        items[f"{full_key}.{i}"] = elem
            else:
                items[full_key] = " | ".join(str(x) for x in v) if v else None
        else:
            items[full_key] = v
    return items


def collect_all_keys(parsed_series):
    """Ordered list of flattened keys seen across all parsed dicts (first-seen order)."""
    seen = {}
    for parsed in parsed_series:
        if not isinstance(parsed, dict):
            continue
        for k in flatten_dict(parsed).keys():
            if k not in seen:
                seen[k] = True
    return list(seen.keys())


def collect_values_for_key(parsed_series, key):
    vals = []
    for parsed in parsed_series:
        if not isinstance(parsed, dict):
            continue
        v = flatten_dict(parsed).get(key)
        if v is not None and v != "":
            vals.append(str(v).strip())
    return vals


# -- column-type classification ------------------------------------------------

def _is_free_text_or_confidence(leaf_name):
    if leaf_name in NO_MANUAL_LEAF_NAMES:
        return True
    return any(leaf_name.endswith(suf) for suf in FREE_TEXT_SUFFIXES)


def _looks_numeric(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def classify_column(key, values):
    """
    Return one of: 'numeric', 'ordinal', 'boolean', 'categorical', 'free_text'.
    Only the first four get a `manual_<key>` column.
    """
    leaf = key.split(".")[-1].lower()
    if _is_free_text_or_confidence(leaf):
        return "free_text"
    if not values:
        return "free_text"

    numeric_count = sum(1 for v in values if _looks_numeric(v))
    if numeric_count / len(values) >= 0.8:
        return "numeric"

    lower_vals = {v.lower().strip() for v in values}

    if lower_vals <= {"yes", "no"} or lower_vals <= {"true", "false"}:
        return "boolean"

    for oset in ORDINAL_SETS:
        if lower_vals <= oset:
            return "ordinal"

    if len(lower_vals) <= MAX_CATEGORICAL_UNIQUE and all(len(v) <= 40 for v in values):
        return "categorical"

    return "free_text"


# -- reformat ------------------------------------------------------------------

def reformat(dest_dir):
    csv_files = sorted(glob.glob(os.path.join(dest_dir, "*.csv")))
    print(f"Found {len(csv_files)} CSV files to reformat.\n")

    for path in csv_files:
        filename = os.path.basename(path)
        df = pd.read_csv(path, low_memory=False)

        if "answer" not in df.columns:
            print(f"  [SKIP] {filename}: no 'answer' column found.\n")
            continue

        parsed_series = df["answer"].apply(extract_json)
        parse_success = parsed_series.apply(lambda x: isinstance(x, dict)).sum()

        # Non-JSON answer files (e.g. the yes/no main_job output)
        if parse_success == 0:
            df["manual_TF"] = ""
            df.to_csv(path, index=False)
            print(f"  Saved: {filename}  ({len(df)} rows, {len(df.columns)} cols)")
            print(f"    No JSON answers - kept original columns + manual_TF.\n")
            continue

        all_keys = collect_all_keys(parsed_series)
        if not all_keys:
            df["manual_TF"] = ""
            df.to_csv(path, index=False)
            print(f"  [WARN] {filename}: JSON parsed but no keys found.\n")
            continue

        flat_series = parsed_series.apply(
            lambda d: flatten_dict(d) if isinstance(d, dict) else {}
        )
        for key in all_keys:
            df[key] = flat_series.apply(lambda f, k=key: f.get(k))

        # one overall-correctness column for the annotator
        df["manual_TF"] = ""

        # one manual_<key> column per labellable JSON key
        added_manual = []
        for key in all_keys:
            values = collect_values_for_key(parsed_series, key)
            col_type = classify_column(key, values)
            if col_type in ("numeric", "ordinal", "boolean", "categorical"):
                manual_col = f"manual_{key.replace('.', '_')}"
                df[manual_col] = ""
                added_manual.append((manual_col, col_type))

        df.to_csv(path, index=False)

        lines = [f"  Saved: {filename}  ({len(df)} rows, {len(df.columns)} cols)"]
        lines.append(f"    JSON parse: {parse_success}/{len(df)} rows")
        lines.append(f"    Keys found:  {all_keys}")
        if added_manual:
            lines.append(f"    Manual review columns added:")
            for mc, ct in added_manual:
                lines.append(f"      {mc}  [{ct}]")
        else:
            lines.append(f"    No labellable fields detected - only manual_TF added.")
        print("\n".join(lines))
        print()

    print(f"Done. Reformed files saved to: {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", default=DEFAULT_RAW)
    parser.add_argument("--dest_dir", default=DEFAULT_DEST)
    args = parser.parse_args()

    stage_repaired(args.raw_dir, args.dest_dir)
    reformat(args.dest_dir)


if __name__ == "__main__":
    main()
