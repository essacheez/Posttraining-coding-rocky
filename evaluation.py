"""Benchmark SFT checkpoints with vLLM for fast batched generation.

Add a benchmark by writing a run_xxx(llm, tokenizer, adapter_path) function that
returns a score, then adding it to BENCHMARK_FUNCTIONS and BENCHMARKS below.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from train_sft import LORA_RANK, MODEL_ID, OUTPUT_DIR

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
CHECKPOINT_GLOB = os.path.join(OUTPUT_DIR, "checkpoint-*")
FINAL_ADAPTER = os.path.join(OUTPUT_DIR, "final_adapter")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "eval_results.json")

MAX_NEW_TOKENS = 512
EXECUTION_TIMEOUT_SECONDS = 10

BENCHMARKS = ["humaneval", "mbpp"]


def find_checkpoints():
    checkpoints = sorted(
        glob.glob(CHECKPOINT_GLOB), key=lambda path: int(path.split("-")[-1])
    )
    if os.path.isdir(FINAL_ADAPTER):
        checkpoints.append(FINAL_ADAPTER)
    return checkpoints


def build_prompt(tokenizer, instruction):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def extract_code(completion):
    match = re.search(r"```(?:python)?\n(.*?)```", completion, re.DOTALL)
    return match.group(1) if match else completion


def execute_code(code):
    # runs model-generated code locally with a timeout, same approach the standard HumanEval harness uses
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            timeout=EXECUTION_TIMEOUT_SECONDS,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False


def generate_completions(llm, tokenizer, prompts, adapter_path):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    lora_request = LoRARequest("adapter", 1, adapter_path) if adapter_path else None
    generation_start = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    generation_seconds = time.time() - generation_start
    print(f"generated {len(prompts)} completions in {generation_seconds:.1f}s")
    return [output.outputs[0].text for output in outputs]


def run_humaneval(llm, tokenizer, adapter_path, num_problems):
    dataset = load_dataset("openai/openai_humaneval", split="test")
    if num_problems:
        dataset = dataset.select(range(num_problems))

    instructions = [
        f"Complete the following function. Reply with only a python code block.\n\n{example['prompt']}"
        for example in dataset
    ]
    prompts = [build_prompt(tokenizer, instruction) for instruction in instructions]
    completions = generate_completions(llm, tokenizer, prompts, adapter_path)

    passed = 0
    for example, completion in zip(dataset, completions):
        code = extract_code(completion)
        full_program = f"{code}\n{example['test']}\ncheck({example['entry_point']})\n"
        if execute_code(full_program):
            passed += 1
    return passed / len(dataset)


def run_mbpp(llm, tokenizer, adapter_path, num_problems):
    dataset = load_dataset("google-research-datasets/mbpp", split="test")
    if num_problems:
        dataset = dataset.select(range(num_problems))

    instructions = [
        f"Write a python function for this task. Reply with only a python code block.\n\n"
        f"{example['text']}\n\nIt must pass these checks:\n"
        + "\n".join(example["test_list"])
        for example in dataset
    ]
    prompts = [build_prompt(tokenizer, instruction) for instruction in instructions]
    completions = generate_completions(llm, tokenizer, prompts, adapter_path)

    passed = 0
    for example, completion in zip(dataset, completions):
        code = extract_code(completion)
        full_program = f"{code}\n" + "\n".join(example["test_list"])
        if execute_code(full_program):
            passed += 1
    return passed / len(dataset)


BENCHMARK_FUNCTIONS = {
    "humaneval": run_humaneval,
    "mbpp": run_mbpp,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark SFT checkpoints with vLLM")
    parser.add_argument(
        "--base",
        action="store_true",
        help="also evaluate the base model with no adapter",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="adapter dirs to evaluate, defaults to all checkpoints under outputs/sft plus final_adapter",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=list(BENCHMARK_FUNCTIONS),
        default=BENCHMARKS,
        help="benchmarks to run, defaults to all",
    )
    parser.add_argument(
        "--num-problems",
        type=int,
        default=None,
        help="subsample each benchmark to this many problems, defaults to the full set",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoints = []
    if args.base:
        checkpoints.append(("base", None))
    if args.checkpoints:
        checkpoints += [
            (os.path.basename(path.rstrip("/")), path) for path in args.checkpoints
        ]
    elif not args.base:
        checkpoints += [(os.path.basename(path), path) for path in find_checkpoints()]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID, enable_lora=True, max_lora_rank=LORA_RANK, dtype="bfloat16"
    )

    results = {}
    for checkpoint_name, adapter_path in checkpoints:
        results[checkpoint_name] = {}
        for benchmark_name in args.benchmarks:
            benchmark_start = time.time()
            score = BENCHMARK_FUNCTIONS[benchmark_name](
                llm, tokenizer, adapter_path, args.num_problems
            )
            benchmark_seconds = time.time() - benchmark_start
            results[checkpoint_name][benchmark_name] = score
            print(
                f"{checkpoint_name} {benchmark_name}: {score:.3f} ({benchmark_seconds:.1f}s total)"
            )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as results_out:
        json.dump(results, results_out, indent=2)
    print(f"wrote results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
