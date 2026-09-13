"""Splits master_dataset_rocky.jsonl into a train set and a single-turn only test set."""

import json
import random
from pathlib import Path

SEED = 42
TEST_SIZE = 200

INPUT_PATH = Path(__file__).parent / "raw" / "master_dataset_rocky.jsonl"
TRAIN_PATH = Path(__file__).parent / "raw" / "train_rocky.jsonl"
TEST_PATH = Path(__file__).parent / "raw" / "test_rocky.jsonl"


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    rng = random.Random(SEED)
    records = [json.loads(line) for line in INPUT_PATH.open(encoding="utf-8")]
    rng.shuffle(records)

    test_records = []
    train_records = []
    for record in records:
        is_single_turn = len(record["messages"]) == 2
        if is_single_turn and len(test_records) < TEST_SIZE:
            test_records.append(record)
        else:
            train_records.append(record)

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    write_jsonl(TRAIN_PATH, train_records)
    write_jsonl(TEST_PATH, test_records)

    print(f"train: {len(train_records)} records ({sum(len(record['messages']) > 2 for record in train_records)} multi-turn)")
    print(f"test: {len(test_records)} records, all single-turn: {all(len(record['messages']) == 2 for record in test_records)}")


if __name__ == "__main__":
    main()
