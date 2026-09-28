"""Build the RLVR GRPO dataset: MBPP problems with executable tests plus a slice of RLAIF persona prompts.
Every MBPP split except test is used, so the MBPP test split stays held out for the STEM eval."""

import json
import os
import random

from datasets import load_dataset

RLVR_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(RLVR_DATA_DIR))
RLAIF_DATASET_FILE = os.path.join(REPO_ROOT, "RLAIF", "data", "grpo_dataset.jsonl")
OUTPUT_FILE = os.path.join(RLVR_DATA_DIR, "rlvr_dataset.jsonl")

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_TRAIN_SPLITS = ["train", "validation", "prompt"]
# persona prompts keep the voice under pressure from the verifier, at this share of the mix
PERSONA_FRACTION = 0.3
SEED = 0


def mbpp_instruction(example):
    # same wording as run_mbpp in EVALS/evaluation.py, so the trained format is the evaluated one
    return (
        "Write a python function for this task. Reply with only a python code block.\n\n"
        f"{example['text']}\n\nIt must pass these checks:\n" + "\n".join(example["test_list"])
    )


def stem_row(example, split):
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": mbpp_instruction(example)}],
        "completion": [{"role": "assistant", "content": f"```python\n{example['code']}\n```"}],
        "kind": "stem",
        "tests": example["test_list"],
        "setup": example["test_setup_code"],
        "source": f"mbpp/{split}/{example['task_id']}",
    }


def persona_row(record):
    return {
        "prompt": record["prompt"],
        "completion": record["completion"],
        "kind": "persona",
        "tests": [],
        "setup": "",
        "source": "rlaif/grpo_dataset",
    }


def load_persona_rows():
    rows = []
    with open(RLAIF_DATASET_FILE, encoding="utf-8") as rlaif_file:
        for line in rlaif_file:
            record = json.loads(line)
            # the RLAIF coding rows are MBPP validation problems, already included here with their tests
            if "python code block" in record["prompt"][-1]["content"]:
                continue
            rows.append(persona_row(record))
    return rows


def main():
    random_generator = random.Random(SEED)

    stem_rows = []
    for split in MBPP_TRAIN_SPLITS:
        for example in load_dataset(MBPP_DATASET, "full", split=split):
            stem_rows.append(stem_row(example, split))

    test_task_ids = {example["task_id"] for example in load_dataset(MBPP_DATASET, "full", split="test")}
    train_task_ids = {int(row["source"].split("/")[-1]) for row in stem_rows}
    overlap = test_task_ids & train_task_ids
    if overlap:
        raise SystemExit(f"{len(overlap)} MBPP test problems leaked into the training rows")

    persona_rows = load_persona_rows()
    random_generator.shuffle(persona_rows)
    persona_count = round(len(stem_rows) * PERSONA_FRACTION / (1 - PERSONA_FRACTION))
    rows = stem_rows + persona_rows[:persona_count]
    random_generator.shuffle(rows)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {OUTPUT_FILE} ({len(stem_rows)} stem, {persona_count} persona)")


if __name__ == "__main__":
    main()
