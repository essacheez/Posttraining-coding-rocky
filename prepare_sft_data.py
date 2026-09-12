import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILES = [
    os.path.join(PROJECT_DIR, "data", "raw", "coding.jsonl"),
    os.path.join(PROJECT_DIR, "data", "raw", "persona.jsonl"),
]
OUTPUT_FILE = os.path.join(PROJECT_DIR, "data", "sft.parquet")
SEED = 0
with open(os.path.join(PROJECT_DIR, "configs", "common.yaml"), encoding="utf-8") as config_file:
    SYSTEM_PROMPT = yaml.safe_load(config_file)["system_prompt"]


def read_jsonl(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        return [json.loads(raw_line) for raw_line in jsonl_file if raw_line.strip()]


def to_messages(record):
    if "messages" in record:
        messages = [dict(message) for message in record["messages"]]
    else:
        messages = [
            {"role": "user", "content": record["prompt"]},
            {"role": "assistant", "content": record["response"]},
        ]
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    assert messages[-1]["role"] == "assistant", "last SFT turn must be an assistant message"
    return messages


def main():
    conversations = []
    for jsonl_path in INPUT_FILES:
        for record in read_jsonl(jsonl_path):
            conversations.append(to_messages(record))
    random.Random(SEED).shuffle(conversations)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    pq.write_table(pa.table({"messages": conversations}), OUTPUT_FILE)
    print(f"wrote {len(conversations)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
