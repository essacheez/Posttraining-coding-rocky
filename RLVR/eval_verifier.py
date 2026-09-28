"""STEM capability on the held out MBPP test split, scored by the RLVR verifier itself.
Reports strict pass@1 and the mean fraction of asserts passed, and saves completions for a hack check."""

import argparse
import json
import os
import sys

RLVR_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(RLVR_DIR)
EVALS_DIR = os.path.join(REPO_ROOT, "EVALS")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EVALS_DIR)

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM

# imported, not copied, so prompts and decoding match the coding benchmarks exactly
from evaluation import build_prompt, generate_completions
from RLVR.training.verifier_mbpp import EQUALITY_OVERRIDE, EXIT_CALL, extract_code, verify_all
from SFT.training.train_sft import LORA_RANK, MODEL_ID

MBPP_SPLIT = "test"
DEFAULT_RESULTS_DIR = os.path.join(EVALS_DIR, "results_rlvr")
RESULTS_NAME = "verifier_results.json"
COMPLETIONS_NAME = "verifier_completions.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Score the held out MBPP test split with the RLVR verifier")
    parser.add_argument("--base", action="store_true", help="evaluate the base model with no adapter")
    parser.add_argument("--checkpoints", nargs="+", default=None, help="adapter dirs to evaluate")
    parser.add_argument("--base-model", default=MODEL_ID, help="base model vLLM loads before the adapter")
    parser.add_argument("--num-problems", type=int, default=None, help="score only the first N problems")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="where the results json goes")
    return parser.parse_args()


def load_problems(num_problems):
    """The MBPP test split, which build_rlvr_dataset.py excluded from training."""
    dataset = load_dataset("google-research-datasets/mbpp", split=MBPP_SPLIT)
    if num_problems:
        dataset = dataset.select(range(num_problems))
    instructions = []
    for example in dataset:
        checks = "\n".join(example["test_list"])
        instructions.append(
            f"Write a python function for this task. Reply with only a python code block.\n\n"
            f"{example['text']}\n\nIt must pass these checks:\n{checks}"
        )
    tests = [list(example["test_list"]) for example in dataset]
    return instructions, tests


def count_hacks(completions):
    """Completions whose code exits early or overrides equality, the two ways the harness can be fooled."""
    flagged = 0
    for completion in completions:
        code = extract_code(completion)
        if EXIT_CALL.search(code) or EQUALITY_OVERRIDE.search(code):
            flagged += 1
    return flagged


def score_stage(llm, tokenizer, adapter_path, instructions, tests):
    prompts = [build_prompt(tokenizer, instruction) for instruction in instructions]
    completions = generate_completions(llm, tokenizer, prompts, adapter_path)
    fractions = verify_all(completions, tests, [""] * len(tests))
    solved = sum(1 for fraction in fractions if fraction == 1.0)
    summary = {
        "verifier_pass_at_1": solved / len(fractions),
        "verifier_assert_fraction": sum(fractions) / len(fractions),
        "problems": len(fractions),
        "hack_flagged": count_hacks(completions),
    }
    return summary, completions


def main():
    args = parse_args()
    checkpoints = []
    if args.base:
        checkpoints.append(("base", None))
    if args.checkpoints:
        checkpoints += [(os.path.basename(path.rstrip("/")), path) for path in args.checkpoints]
    if not checkpoints:
        raise SystemExit("pass --base or --checkpoints")

    instructions, tests = load_problems(args.num_problems)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    llm = LLM(model=args.base_model, enable_lora=True, max_lora_rank=LORA_RANK, dtype="bfloat16")

    results = {}
    generations = {}
    for name, adapter_path in checkpoints:
        summary, completions = score_stage(llm, tokenizer, adapter_path, instructions, tests)
        results[name] = summary
        generations[name] = completions
        print(f"{name}: pass@1 {summary['verifier_pass_at_1']:.3f}, asserts {summary['verifier_assert_fraction']:.3f}, hacks {summary['hack_flagged']}")

    os.makedirs(args.results_dir, exist_ok=True)
    results_path = os.path.join(args.results_dir, RESULTS_NAME)
    existing = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as results_file:
            existing = json.load(results_file)
    existing.update(results)
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump(existing, results_file, indent=2)
    completions_path = os.path.join(args.results_dir, COMPLETIONS_NAME)
    with open(completions_path, "w", encoding="utf-8") as completions_file:
        json.dump(generations, completions_file, indent=2)
    print(f"wrote {results_path}\nwrote {completions_path}")


if __name__ == "__main__":
    main()
