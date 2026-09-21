"""Print held-out prompts with the human reference, SFT output and RLAIF output as markdown."""
import argparse
import json
import os

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVALS_DIR)
TEST_FILE = os.path.join(REPO_ROOT, "SFT", "data", "raw", "test_rocky.jsonl")
GENERATIONS = os.path.join(EVALS_DIR, "results_rlaif", "persona_generations.json")
DEFAULT_INDICES = [10, 137, 155]


def quote(text):
    return "\n".join("> " + line for line in text.strip().split("\n"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", default=GENERATIONS)
    parser.add_argument("--sft-key", default="base (sft_merged)")
    parser.add_argument("--rlaif-key", default="RLAIF_model")
    parser.add_argument("--indices", type=int, nargs="+", default=DEFAULT_INDICES)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(TEST_FILE, encoding="utf-8")]
    generations = json.load(open(args.generations))
    sft_outputs = generations[args.sft_key]
    rlaif_outputs = generations[args.rlaif_key]

    for index in args.indices:
        messages = rows[index]["messages"]
        prompt = [message["content"] for message in messages if message["role"] == "user"][0]
        reference = [message["content"] for message in messages if message["role"] == "assistant"][0]
        print(f"**Prompt {index}:** {prompt.strip()}\n")
        print("Human reference:\n")
        print(quote(reference) + "\n")
        print("SFT:\n")
        print(quote(sft_outputs[index]) + "\n")
        print("RLAIF_model:\n")
        print(quote(rlaif_outputs[index]) + "\n")


if __name__ == "__main__":
    main()
