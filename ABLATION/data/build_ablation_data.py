"""Build the ablation datasets: the SFT and RLAIF mixes rebuilt with MBPP coding rows swapped in.
Total row counts match the originals, so composition is the only variable that changes."""

import json
import os
import random
import sys

ABLATION_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.dirname(ABLATION_DATA_DIR)
REPO_ROOT = os.path.dirname(ABLATION_DIR)
sys.path.insert(0, REPO_ROOT)

from datasets import load_dataset

from RLVR.training.verifier import verify

SFT_SOURCE = os.path.join(REPO_ROOT, "SFT", "data", "persona_training_data.jsonl")
RLAIF_SOURCE = os.path.join(REPO_ROOT, "RLAIF", "data", "grpo_dataset.jsonl")
SFT_OUTPUT = os.path.join(ABLATION_DATA_DIR, "sft_with_code.jsonl")
RLAIF_OUTPUT = os.path.join(ABLATION_DATA_DIR, "grpo_with_code.jsonl")

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_TRAIN_SPLITS = ["train", "validation", "prompt"]

# a quarter of each mix becomes coding, replacing persona rows so the totals stay 1000 and 600
SFT_TOTAL = 1000
RLAIF_TOTAL = 600
CODING_FRACTION = 0.25
SEED = 0


def mbpp_instruction(example):
    # same wording as run_mbpp in EVALS/evaluation.py, so the trained format is the evaluated one
    return (
        "Write a python function for this task. Reply with only a python code block.\n\n"
        f"{example['text']}\n\nIt must pass these checks:\n" + "\n".join(example["test_list"])
    )


def coding_row(example, split):
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": mbpp_instruction(example)}],
        "completion": [{"role": "assistant", "content": f"```python\n{example['code']}\n```"}],
        "source": f"mbpp/{split}/{example['task_id']}",
    }


def load_mbpp_pool():
    """Every MBPP split except test, so the 500 problem test split stays held out."""
    pool = []
    for split in MBPP_TRAIN_SPLITS:
        for example in load_dataset(MBPP_DATASET, "full", split=split):
            pool.append((example, split))
    return pool


def read_jsonl(path):
    with open(path, encoding="utf-8") as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as jsonl_file:
        for row in rows:
            jsonl_file.write(json.dumps(row) + "\n")


def check_no_test_leak(rows):
    test_ids = {example["task_id"] for example in load_dataset(MBPP_DATASET, "full", split="test")}
    for row in rows:
        if row.get("source", "").startswith("mbpp/"):
            if int(row["source"].split("/")[-1]) in test_ids:
                raise SystemExit(f"MBPP test problem leaked into the mix: {row['source']}")


def check_solutions_pass(rows):
    """Every coding target must pass its own asserts, so the mix teaches correct code only."""
    failed = []
    for row, example in rows:
        score = verify(row["completion"][-1]["content"], example["test_list"], example["test_setup_code"])
        if score < 1.0:
            failed.append((row["source"], score))
    return failed


def main():
    random_generator = random.Random(SEED)
    pool = load_mbpp_pool()
    random_generator.shuffle(pool)

    sft_coding_count = int(SFT_TOTAL * CODING_FRACTION)
    rlaif_coding_count = int(RLAIF_TOTAL * CODING_FRACTION)
    if sft_coding_count + rlaif_coding_count > len(pool):
        raise SystemExit(f"need {sft_coding_count + rlaif_coding_count} MBPP problems, pool has {len(pool)}")

    # disjoint slices, so the RLAIF stage never re-sees a problem the SFT stage trained on
    sft_slice = pool[:sft_coding_count]
    rlaif_slice = pool[sft_coding_count : sft_coding_count + rlaif_coding_count]
    sft_coding = [(coding_row(example, split), example) for example, split in sft_slice]
    rlaif_coding = [(coding_row(example, split), example) for example, split in rlaif_slice]

    failed = check_solutions_pass(sft_coding + rlaif_coding)
    if failed:
        raise SystemExit(f"{len(failed)} MBPP reference solutions failed their own asserts: {failed[:5]}")

    persona_rows = read_jsonl(SFT_SOURCE)
    random_generator.shuffle(persona_rows)
    sft_rows = persona_rows[: SFT_TOTAL - sft_coding_count] + [row for row, example in sft_coding]

    rlaif_source_rows = read_jsonl(RLAIF_SOURCE)
    # drop the coding rows already in the RLAIF mix so the new share is exactly CODING_FRACTION
    rlaif_persona = [row for row in rlaif_source_rows if "python code block" not in row["prompt"][-1]["content"]]
    random_generator.shuffle(rlaif_persona)
    rlaif_rows = rlaif_persona[: RLAIF_TOTAL - rlaif_coding_count] + [row for row, example in rlaif_coding]

    check_no_test_leak(sft_rows + rlaif_rows)
    random_generator.shuffle(sft_rows)
    random_generator.shuffle(rlaif_rows)
    write_jsonl(SFT_OUTPUT, sft_rows)
    write_jsonl(RLAIF_OUTPUT, rlaif_rows)
    print(f"wrote {len(sft_rows)} rows ({sft_coding_count} coding) to {SFT_OUTPUT}")
    print(f"wrote {len(rlaif_rows)} rows ({rlaif_coding_count} coding) to {RLAIF_OUTPUT}")
    print(f"MBPP problems used: {sft_coding_count} for SFT, {rlaif_coding_count} for RLAIF, disjoint, none from the test split")


if __name__ == "__main__":
    main()
