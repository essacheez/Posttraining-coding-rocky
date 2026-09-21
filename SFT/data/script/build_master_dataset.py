"""Builds a master chat dataset from No Robots, Dolly 15k and OASST2 and writes it as JSONL."""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset

SEED = 42
OUTPUT_PATH = Path(__file__).parent / "master_dataset.jsonl"

NO_ROBOTS_QUOTAS = {
    "Generation": 200,
    "Open QA": 120,
    "Brainstorm": 100,
    "Coding": 70,
    "Rewrite": 70,
    "Summarize": 50,
    "Closed QA": 40,
    "Classify": 30,
    "Extract": 20,
}

DOLLY_QUOTAS = {
    "open_qa": 70,
    "closed_qa": 60,
    "information_extraction": 50,
    "summarization": 40,
    "classification": 40,
    "brainstorming": 40,
    "general_qa": 30,
    "creative_writing": 20,
}

OASST_QUOTA = 150
OASST_MIN_MESSAGES = 4
UNRANKED = 999


def sample_by_category(rows, quotas, category_key, rng):
    """Returns a random sample of rows per category matching the quotas."""
    grouped = defaultdict(list)
    for row in rows:
        if row[category_key] in quotas:
            grouped[row[category_key]].append(row)

    sampled = []
    for category, quota in quotas.items():
        sampled.extend(rng.sample(grouped[category], quota))
    return sampled


def build_no_robots(rng):
    rows = load_dataset("HuggingFaceH4/no_robots", split="train")
    sampled = sample_by_category(rows, NO_ROBOTS_QUOTAS, "category", rng)

    records = []
    for row in sampled:
        messages = [{"role": message["role"], "content": message["content"]} for message in row["messages"]]
        records.append({
            "id": f"no_robots/{row['prompt_id']}",
            "source": "no_robots",
            "category": row["category"],
            "num_turns": len(messages),
            "messages": messages,
        })
    return records


def build_dolly(rng):
    rows = load_dataset("databricks/databricks-dolly-15k", split="train")
    indexed_rows = [dict(row, row_index=row_index) for row_index, row in enumerate(rows)]
    sampled = sample_by_category(indexed_rows, DOLLY_QUOTAS, "category", rng)

    records = []
    for row in sampled:
        user_content = row["instruction"].strip()
        if row["context"].strip():
            user_content = f"{user_content}\n\n{row['context'].strip()}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": row["response"].strip()},
        ]
        records.append({
            "id": f"dolly/{row['row_index']}",
            "source": "dolly",
            "category": row["category"],
            "num_turns": len(messages),
            "messages": messages,
        })
    return records


def quality_label(message):
    """Returns the crowd quality score of an OASST message, 0 when missing."""
    labels = message["labels"]
    if labels is None:
        return 0.0
    for name, value in zip(labels["name"], labels["value"]):
        if name == "quality":
            return value
    return 0.0


def assistant_sort_key(message):
    rank = message["rank"] if message["rank"] is not None else UNRANKED
    return (rank, -quality_label(message))


def subtree_depth(message_id, children, depth_cache):
    """Longest chain of messages starting at message_id, counting itself."""
    if message_id in depth_cache:
        return depth_cache[message_id]
    child_depths = [subtree_depth(child["message_id"], children, depth_cache) for child in children[message_id]]
    depth_cache[message_id] = 1 + max(child_depths, default=0)
    return depth_cache[message_id]


def best_path(root, children, depth_cache):
    """Walks a tree taking the best-ranked assistant reply and the deepest prompter follow-up."""
    path = [root]
    current = root
    while children[current["message_id"]]:
        candidates = children[current["message_id"]]
        if current["role"] == "prompter":
            current = min(candidates, key=assistant_sort_key)
        else:
            current = max(candidates, key=lambda child: subtree_depth(child["message_id"], children, depth_cache))
        path.append(current)

    while path and path[-1]["role"] != "assistant":
        path.pop()
    return path


def build_oasst(rng):
    rows = load_dataset("OpenAssistant/oasst2", split="train")
    usable = [
        row for row in rows
        if row["lang"] == "en" and not row["deleted"] and row["review_result"] is not False
    ]

    children = defaultdict(list)
    roots = []
    for row in usable:
        if row["parent_id"] is None:
            roots.append(row)
        else:
            children[row["parent_id"]].append(row)

    depth_cache = {}
    conversations = []
    for root in roots:
        path = best_path(root, children, depth_cache)
        if len(path) >= OASST_MIN_MESSAGES:
            conversations.append(path)

    print(f"oasst2: {len(roots)} english trees, {len(conversations)} paths with {OASST_MIN_MESSAGES}+ messages")
    sampled = rng.sample(conversations, OASST_QUOTA)

    records = []
    for path in sampled:
        messages = []
        for message in path:
            role = "user" if message["role"] == "prompter" else "assistant"
            messages.append({"role": role, "content": message["text"].strip()})
        records.append({
            "id": f"oasst2/{path[0]['message_tree_id']}",
            "source": "oasst2",
            "category": "multi_turn",
            "num_turns": len(messages),
            "messages": messages,
        })
    return records


def main():
    rng = random.Random(SEED)
    records = build_no_robots(rng) + build_dolly(rng) + build_oasst(rng)
    rng.shuffle(records)

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps({"messages": record["messages"]}, ensure_ascii=False) + "\n")

    print(f"wrote {len(records)} records to {OUTPUT_PATH}")
    print("per source:", dict(Counter(record["source"] for record in records)))
    print("per category:", dict(Counter((record["source"], record["category"]) for record in records)))
    print("turn counts:", dict(sorted(Counter(record["num_turns"] for record in records).items())))


if __name__ == "__main__":
    main()
