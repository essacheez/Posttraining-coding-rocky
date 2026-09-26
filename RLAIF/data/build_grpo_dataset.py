import json
import os
import random

from datasets import load_dataset

RLAIF_DATA_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(RLAIF_DATA_DIR))
ASSISTANT_FILE = os.path.join(RLAIF_DATA_DIR, "assistant_prompts.jsonl")
SFT_TRAIN_FILE = os.path.join(REPO_ROOT, "SFT", "data", "raw", "train_rocky.jsonl")
OUTPUT_FILE = os.path.join(RLAIF_DATA_DIR, "grpo_dataset.jsonl")

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
NARRATIVE_KEYWORDS = ["write a letter", "write a story", "short story", "write a poem", "describe", "diary", "tell me a story", "write an email", "narrate", "essay"]
NUM_NARRATIVE = 60
NUM_CODING = 40
SEED = 0


def make_row(question, answer):
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        "completion": [{"role": "assistant", "content": answer}],
    }


def main():
    random.seed(SEED)
    rows = [json.loads(line) for line in open(ASSISTANT_FILE, encoding="utf-8")]

    narrative = []
    for line in open(SFT_TRAIN_FILE, encoding="utf-8"):
        messages = json.loads(line)["messages"]
        question = messages[0]["content"]
        answer = messages[-1]["content"]
        lowered = question.lower()
        if "```" in answer or "python" in lowered or "code" in lowered:
            continue
        if any(keyword in lowered for keyword in NARRATIVE_KEYWORDS):
            narrative.append(make_row(question, answer))
    random.shuffle(narrative)
    rows += narrative[:NUM_NARRATIVE]

    # MBPP validation split is disjoint from the test split the coding eval uses
    mbpp = load_dataset("google-research-datasets/mbpp", split="validation")
    coding = []
    for example in mbpp:
        question = f"{example['text']} Reply with only a python code block."
        answer = f"```python\n{example['code']}\n```"
        coding.append(make_row(question, answer))
    random.shuffle(coding)
    rows += coding[:NUM_CODING]

    random.shuffle(rows)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(f"wrote {len(rows)} rows ({len(narrative[:NUM_NARRATIVE])} narrative, {len(coding[:NUM_CODING])} coding)")


if __name__ == "__main__":
    main()
