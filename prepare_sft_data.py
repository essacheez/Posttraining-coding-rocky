import json
import os
import random

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# test_rocky.jsonl is held out for evaluate_persona.py and stays out of training
INPUT_FILES = [
    os.path.join(PROJECT_DIR, "data", "train_rocky.jsonl"),
]

OUTPUT_FILE = os.path.join(PROJECT_DIR, "data", "sft.jsonl")
SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
SEED = 0


def read_jsonl(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        return [json.loads(raw_line) for raw_line in jsonl_file if raw_line.strip()]


def to_prompt_completion(record):
    """Prompt is every turn but the last, completion is the final assistant turn. Loss is computed on the completion only."""
    if "messages" in record:
        messages = [dict(message) for message in record["messages"]]
    else:
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    assert messages[-1]["role"] == "assistant", "last turn must be an assistant message"
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def main():
    rows = []
    for jsonl_path in INPUT_FILES:
        for record in read_jsonl(jsonl_path):
            rows.append(to_prompt_completion(record))
    random.Random(SEED).shuffle(rows)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as output_file:
        for row in rows:
            output_file.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
