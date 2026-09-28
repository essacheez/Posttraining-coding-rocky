"""Build the SFT mix whose coding rows are the base model's own verified correct solutions.
Sampling from the base model keeps the targets in its idiom, so imitating them costs no style."""

import json
import os
import sys

ABLATION_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.dirname(ABLATION_DATA_DIR)
REPO_ROOT = os.path.dirname(ABLATION_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "EVALS"))

import random

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from evaluation import build_prompt
from RLVR.training.verifier import verify
from SFT.training.train_sft import MODEL_ID

SFT_SOURCE = os.path.join(REPO_ROOT, "SFT", "data", "persona_training_data.jsonl")
OUTPUT_FILE = os.path.join(ABLATION_DATA_DIR, "sft_with_selfgen.jsonl")
SOLUTIONS_FILE = os.path.join(ABLATION_DATA_DIR, "selfgen_solutions.json")

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_TRAIN_SPLITS = ["train", "validation", "prompt"]

SFT_TOTAL = 1000
CODING_COUNT = 250
SAMPLES_PER_PROBLEM = 8
TEMPERATURE = 0.8
TOP_P = 0.95
MAX_NEW_TOKENS = 512
SEED = 0


def mbpp_instruction(example):
    return (
        "Write a python function for this task. Reply with only a python code block.\n\n"
        f"{example['text']}\n\nIt must pass these checks:\n" + "\n".join(example["test_list"])
    )


def load_pool():
    pool = []
    for split in MBPP_TRAIN_SPLITS:
        for example in load_dataset(MBPP_DATASET, "full", split=split):
            pool.append((example, split))
    return pool


def read_jsonl(path):
    with open(path, encoding="utf-8") as jsonl_file:
        return [json.loads(line) for line in jsonl_file if line.strip()]


def pick_correct_solutions(pool):
    """Sample from the base model and keep the first completion that passes every assert."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = [build_prompt(tokenizer, mbpp_instruction(example)) for example, split in pool]
    sampling = SamplingParams(n=SAMPLES_PER_PROBLEM, temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS, seed=SEED)
    llm = LLM(model=MODEL_ID, dtype="bfloat16")
    outputs = llm.generate(prompts, sampling)

    solved = []
    for (example, split), output in zip(pool, outputs):
        for candidate in output.outputs:
            if verify(candidate.text, example["test_list"], example["test_setup_code"]) == 1.0:
                solved.append((example, split, candidate.text))
                break
    print(f"{len(solved)} of {len(pool)} problems solved by at least one of {SAMPLES_PER_PROBLEM} samples")
    return solved


def coding_row(example, split, solution_text):
    # keep only the code, re-fenced, so the target is a clean single block like the MBPP rows
    from RLVR.training.verifier import extract_code

    code = extract_code(solution_text).strip()
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": mbpp_instruction(example)}],
        "completion": [{"role": "assistant", "content": f"```python\n{code}\n```"}],
        "source": f"mbpp/{split}/{example['task_id']}",
    }


def main():
    random_generator = random.Random(SEED)
    pool = load_pool()
    test_ids = {example["task_id"] for example in load_dataset(MBPP_DATASET, "full", split="test")}
    for example, split in pool:
        if example["task_id"] in test_ids:
            raise SystemExit(f"MBPP test problem in the training pool: {example['task_id']}")

    solved = pick_correct_solutions(pool)
    if len(solved) < CODING_COUNT:
        raise SystemExit(f"only {len(solved)} solved problems, need {CODING_COUNT}")
    random_generator.shuffle(solved)
    chosen = solved[:CODING_COUNT]
    coding_rows = [coding_row(example, split, text) for example, split, text in chosen]

    for row, (example, split, text) in zip(coding_rows, chosen):
        if verify(row["completion"][-1]["content"], example["test_list"], example["test_setup_code"]) < 1.0:
            raise SystemExit(f"re-fenced target stopped passing: {row['source']}")

    persona_rows = read_jsonl(SFT_SOURCE)
    random_generator.shuffle(persona_rows)
    rows = persona_rows[: SFT_TOTAL - CODING_COUNT] + coding_rows
    random_generator.shuffle(rows)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")
    with open(SOLUTIONS_FILE, "w", encoding="utf-8") as solutions_file:
        json.dump([{"source": row["source"], "code": row["completion"][-1]["content"]} for row in coding_rows], solutions_file, indent=1)
    print(f"wrote {len(rows)} rows ({CODING_COUNT} self generated coding) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
