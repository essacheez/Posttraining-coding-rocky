"""Benchmark SFT checkpoints with vLLM for fast batched generation.

Add a benchmark by writing a run_xxx(llm, tokenizer, adapter_path, num_problems)
function that returns a score, then adding it to BENCHMARK_FUNCTIONS and BENCHMARKS
below. Leave it out of BENCHMARKS to keep it opt-in via --benchmarks.
"""

import argparse
import base64
import glob
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time
import zlib

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVALS_DIR)
sys.path.insert(0, REPO_ROOT)

from SFT.training.train_sft import LORA_RANK, MODEL_ID
from SFT.training.train_sft import OUTPUT_DIR as SFT_MODEL_DIR
from RLAIF.training.train_rlaif import OUTPUT_DIR as RLAIF_MODEL_DIR

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
RESULTS_DIR = os.path.join(EVALS_DIR, "results_rlaif")
CHECKPOINT_GLOB = os.path.join(RLAIF_MODEL_DIR, "checkpoint-*")
FINAL_ADAPTER = os.path.join(RLAIF_MODEL_DIR, "final_adapter")
RESULTS_FILE = os.path.join(RESULTS_DIR, "eval_results_rlaif.json")

MAX_NEW_TOKENS = 1024
EXECUTION_TIMEOUT_SECONDS = 10

# HumanEval+ tests are ~128x larger, and a few take over 10s even when correct
HUMANEVAL_PLUS_TIMEOUT_SECONDS = 60
BIGCODEBENCH_MAX_NEW_TOKENS = 1024
BIGCODEBENCH_TIMEOUT_SECONDS = 60
BIGCODEBENCH_SPLIT = "v0.1.4"

LIVECODEBENCH_REPO = "livecodebench/code_generation_lite"
LIVECODEBENCH_FILE = "test6.jsonl"
LIVECODEBENCH_MAX_NEW_TOKENS = 1024
LIVECODEBENCH_TIMEOUT_SECONDS = 10

# bigcodebench and livecodebench are opt-in: the former needs the extras from pyproject to execute
BENCHMARKS = ["humaneval", "humanevalplus", "mbpp"]
BENCHMARK_DEFAULT_LIMITS = {"bigcodebench": 200}


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


def execute_code(code, timeout=EXECUTION_TIMEOUT_SECONDS):
    # runs model-generated code locally with a timeout, same approach the standard HumanEval harness uses.
    # the program goes through a file rather than `python -c`, since HumanEval+'s tests are large
    # enough to blow past ARG_MAX and raise OSError instead of failing the problem
    with tempfile.TemporaryDirectory() as scratch_dir:
        program_path = os.path.join(scratch_dir, "program.py")
        with open(program_path, "w", encoding="utf-8") as program_file:
            program_file.write(code)
        try:
            result = subprocess.run(
                [sys.executable, program_path],
                capture_output=True,
                timeout=timeout,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False


ADAPTER_IDS = {}


def lora_request_for(adapter_path):
    """vLLM keys its adapter cache on both the name and the int id, so every
    adapter needs its own pair. Reusing one pair across checkpoints makes later
    checkpoints silently serve the first adapter's weights."""
    if not adapter_path:
        return None
    if adapter_path not in ADAPTER_IDS:
        ADAPTER_IDS[adapter_path] = len(ADAPTER_IDS) + 1
    return LoRARequest(adapter_path, ADAPTER_IDS[adapter_path], adapter_path)


def generate_completions(llm, tokenizer, prompts, adapter_path, max_tokens=MAX_NEW_TOKENS):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    lora_request = lora_request_for(adapter_path)
    generation_start = time.time()
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
    generation_seconds = time.time() - generation_start
    print(f"generated {len(prompts)} completions in {generation_seconds:.1f}s")
    return [output.outputs[0].text for output in outputs]


# HumanEval and HumanEval+ pose the same problems, so completions are shared
HUMANEVAL_COMPLETIONS = {}


def run_humaneval_style(llm, tokenizer, adapter_path, num_problems, dataset_name, timeout):
    dataset = load_dataset(dataset_name, split="test")
    if num_problems:
        dataset = dataset.select(range(num_problems))

    instructions = [
        f"Complete the following function. Reply with only a python code block.\n\n{example['prompt']}"
        for example in dataset
    ]
    prompts = [build_prompt(tokenizer, instruction) for instruction in instructions]

    cache_key = adapter_path or "base"
    missing = list(
        dict.fromkeys(
            prompt
            for prompt in prompts
            if (cache_key, prompt) not in HUMANEVAL_COMPLETIONS
        )
    )
    if missing:
        for prompt, completion in zip(
            missing, generate_completions(llm, tokenizer, missing, adapter_path)
        ):
            HUMANEVAL_COMPLETIONS[(cache_key, prompt)] = completion
    completions = [HUMANEVAL_COMPLETIONS[(cache_key, prompt)] for prompt in prompts]

    passed = 0
    for example, completion in zip(dataset, completions):
        code = extract_code(completion)
        full_program = f"{code}\n{example['test']}\ncheck({example['entry_point']})\n"
        if execute_code(full_program, timeout=timeout):
            passed += 1
    return passed / len(dataset)


def run_humaneval(llm, tokenizer, adapter_path, num_problems):
    return run_humaneval_style(
        llm,
        tokenizer,
        adapter_path,
        num_problems,
        "openai/openai_humaneval",
        EXECUTION_TIMEOUT_SECONDS,
    )


def run_humaneval_plus(llm, tokenizer, adapter_path, num_problems):
    return run_humaneval_style(
        llm,
        tokenizer,
        adapter_path,
        num_problems,
        "evalplus/humanevalplus",
        HUMANEVAL_PLUS_TIMEOUT_SECONDS,
    )


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


# its tests write files and draw plots, so isolate the cwd and the plot backend
BIGCODEBENCH_PREAMBLE = """\
import os, tempfile
os.chdir(tempfile.mkdtemp())
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass
"""

# the test column defines TestCases but no runner, and unittest.main() would try
# to parse sys.argv, which is the harness's own program path here
BIGCODEBENCH_RUNNER = """
import sys as _sys, unittest as _unittest

_result = _unittest.TextTestRunner(verbosity=0).run(
    _unittest.TestLoader().loadTestsFromTestCase(TestCases)
)
_sys.exit(0 if _result.wasSuccessful() else 1)
"""


def run_bigcodebench(llm, tokenizer, adapter_path, num_problems):
    dataset = load_dataset("bigcode/bigcodebench", split=BIGCODEBENCH_SPLIT)
    if num_problems:
        dataset = dataset.select(range(num_problems))

    instructions = [
        f"{example['instruct_prompt']}\n\nReply with only a python code block."
        for example in dataset
    ]
    prompts = [build_prompt(tokenizer, instruction) for instruction in instructions]
    completions = generate_completions(
        llm, tokenizer, prompts, adapter_path, max_tokens=BIGCODEBENCH_MAX_NEW_TOKENS
    )

    passed = 0
    for example, completion in zip(dataset, completions):
        code = extract_code(completion)
        # order matters: the tests rely on names the solution imports
        full_program = (
            f"{BIGCODEBENCH_PREAMBLE}\n{code}\n{example['test']}\n{BIGCODEBENCH_RUNNER}"
        )
        if execute_code(full_program, timeout=BIGCODEBENCH_TIMEOUT_SECONDS):
            passed += 1
    return passed / len(dataset)


def load_livecodebench(num_problems):
    """release_v6 ( only, most recently dated problems first"""
    path = hf_hub_download(LIVECODEBENCH_REPO, LIVECODEBENCH_FILE, repo_type="dataset")
    dataset = load_dataset("json", data_files=path, split="train")
    dataset = dataset.sort("contest_date", reverse=True)
    if num_problems:
        dataset = dataset.select(range(min(num_problems, len(dataset))))
    return dataset


def decode_test_cases(example): # private_test_cases is base64+zlib+pickle encoded to resist scraping
    try:
        return json.loads(example["private_test_cases"])
    except json.JSONDecodeError:
        compressed = base64.b64decode(example["private_test_cases"])
        return json.loads(pickle.loads(zlib.decompress(compressed)))


def build_livecodebench_instruction(example):
    if example["starter_code"]:
        return (
            f"{example['question_content']}\n\n"
            "Complete the following function. Reply with only a python code block.\n\n"
            f"{example['starter_code']}"
        )
    return (
        f"{example['question_content']}\n\n"
        "Write a complete python program that reads the input from stdin and "
        "prints the answer to stdout. Reply with only a python code block."
    )


def livecodebench_stdin_program(code, test_case):
    return f"""\
import contextlib, io, sys

sys.stdin = io.StringIO({test_case["input"]!r})
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    exec({code!r}, {{"__name__": "__main__"}})
sys.exit(0 if captured.getvalue().strip() == {test_case["output"]!r}.strip() else 1)
"""


def livecodebench_functional_program(code, test_case, func_name):
    return f"""\
import ast, sys
from typing import *

{code}

args = [ast.literal_eval(line) for line in {test_case["input"]!r}.split("\\n")]
expected = ast.literal_eval({test_case["output"]!r})
sys.exit(0 if Solution().{func_name}(*args) == expected else 1)
"""


def run_livecodebench(llm, tokenizer, adapter_path, num_problems):
    dataset = load_livecodebench(num_problems)

    instructions = [build_livecodebench_instruction(example) for example in dataset]
    prompts = [build_prompt(tokenizer, instruction) for instruction in instructions]
    completions = generate_completions(
        llm, tokenizer, prompts, adapter_path, max_tokens=LIVECODEBENCH_MAX_NEW_TOKENS
    )

    passed = 0
    for example, completion in zip(dataset, completions):
        code = extract_code(completion)
        test_cases = decode_test_cases(example)
        func_name = json.loads(example["metadata"]).get("func_name")

        if all(
            execute_code(
                livecodebench_functional_program(code, test_case, func_name)
                if test_case["testtype"] == "functional"
                else livecodebench_stdin_program(code, test_case),
                timeout=LIVECODEBENCH_TIMEOUT_SECONDS,
            )
            for test_case in test_cases
        ):
            passed += 1
    return passed / len(dataset)


BENCHMARK_FUNCTIONS = {
    "humaneval": run_humaneval,
    "humanevalplus": run_humaneval_plus,
    "mbpp": run_mbpp,
    "bigcodebench": run_bigcodebench,
    "livecodebench": run_livecodebench,
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
        help="adapter dirs to evaluate, defaults to all checkpoints under SFT/model plus final_adapter",
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
        help="subsample each benchmark to this many problems, defaults to the full set "
        "except bigcodebench, which defaults to 200 of 1140",
    )
    parser.add_argument(
        "--base-model",
        default=MODEL_ID,
        help="base model vLLM loads before applying the LoRA adapter, defaults to the "
        "SFT base model; pass the RLAIF merged model dir (RLAIF/model/sft_merged) when "
        "evaluating RLAIF adapters, since those are a delta on top of it, not on MODEL_ID",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="directory to write eval_results.json into, defaults to EVALS/results",
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

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    llm = LLM(
        model=args.base_model, enable_lora=True, max_lora_rank=LORA_RANK, dtype="bfloat16"
    )

    results = {}
    for checkpoint_name, adapter_path in checkpoints:
        results[checkpoint_name] = {}
        for benchmark_name in args.benchmarks:
            benchmark_start = time.time()
            num_problems = args.num_problems or BENCHMARK_DEFAULT_LIMITS.get(
                benchmark_name
            )
            score = BENCHMARK_FUNCTIONS[benchmark_name](
                llm, tokenizer, adapter_path, num_problems
            )
            benchmark_seconds = time.time() - benchmark_start
            results[checkpoint_name][benchmark_name] = score
            print(
                f"{checkpoint_name} {benchmark_name}: {score:.3f} ({benchmark_seconds:.1f}s total)"
            )

    os.makedirs(args.results_dir, exist_ok=True)
    results_file = os.path.join(args.results_dir, "eval_results.json")
    with open(results_file, "w", encoding="utf-8") as results_out:
        json.dump(results, results_out, indent=2)
    print(f"wrote results to {results_file}")


if __name__ == "__main__":
    main()
