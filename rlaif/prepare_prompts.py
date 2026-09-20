"""Turns the SFT training set into a prompts-only file for GRPO rollouts.

RLAIF needs the prompts, not the responses: the policy generates its own
completions and the judge scores them. The held-out 200 in test_rocky.jsonl stay
untouched so the checkpoint 2 eval is comparable to checkpoint 1's.

    uv run python rlaif/prepare_prompts.py
"""

import json
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(PROJECT_DIR, "data", "raw", "train_rocky.jsonl")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "data", "rlaif_prompts.jsonl")

# same system prompt as SFT and both eval harnesses, so the persona has to come
# from the weights rather than from an instruction at rollout time
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def read_jsonl(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        return [json.loads(raw_line) for raw_line in jsonl_file if raw_line.strip()]


def to_prompt(record):
    """Every turn but the final assistant one, under the key TRL expects.

    TRL reads the "prompt" column and applies the chat template itself, so the
    messages are handed over as-is.
    """
    messages = [dict(message) for message in record["messages"]]
    assert messages[-1]["role"] == "assistant", "last turn must be an assistant message"
    messages = messages[:-1]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return {"prompt": messages}


def main():
    rows = [to_prompt(record) for record in read_jsonl(INPUT_FILE)]
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} prompts to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
