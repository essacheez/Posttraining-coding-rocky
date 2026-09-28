"""Gather every cached generation set into one file and compute surface metrics per set."""
import json
import os
import re
import statistics
import sys

WORK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(WORK_DIR)
sys.path.insert(0, os.path.join(REPO, "EVALS"))
from evaluate_persona import load_eval_examples, load_persona_corpus, standardizer, centroid_distance, profile_vector, cosine

collapsed = json.load(open(os.path.join(WORK_DIR, "data", "first_rlaif_attempt", "persona_generations.json")))
final = json.load(open(os.path.join(REPO, "EVALS", "results_rlaif", "persona_generations.json")))
examples = load_eval_examples(None)

sets = {
    "SFT (gen A)": collapsed["base"],
    "v1 step 15": collapsed["checkpoint-15"],
    "v1 step 45": collapsed["checkpoint-45"],
    "v1 step 75": collapsed["checkpoint-75"],
    "v1 step 105": collapsed["checkpoint-105"],
    "v1 step 112": collapsed["checkpoint-112"],
    "SFT (gen B)": final["base (sft_merged)"],
    "v2 final": final["RLAIF_model"],
    "human reference": [example["reference"] for example in examples],
}
prompts = [example["prompt"][-1]["content"] for example in examples]
json.dump({"prompts": prompts, "sets": sets}, open(os.path.join(WORK_DIR, "data", "all_generations.json"), "w"))

corpus = load_persona_corpus()
means, stds = standardizer(corpus)
corpus_profile = profile_vector(corpus)


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def rate(texts, word):
    tokens = [token for text in texts for token in words(text)]
    return 1000 * tokens.count(word) / len(tokens)


summary = {}
for name, texts in sets.items():
    summary[name] = {
        "style_cosine": cosine(profile_vector(texts), corpus_profile),
        "centroid_distance": centroid_distance(texts, means, stds),
        "opens_listen": sum(text.lstrip().lower().startswith("listen") for text in texts) / len(texts),
        "has_listen_listen": sum(bool(re.search(r"(?i)listen\.\s*listen", text)) for text in texts) / len(texts),
        "closes_good_good": sum(bool(re.search(r"(?i)good,\s*good,\s*good\W*$", text.strip())) for text in texts) / len(texts),
        "question_tag": sum(bool(re.search(r"(?i),\s*(yes|question)\?", text)) for text in texts) / len(texts),
        "mean_words": statistics.mean(len(text.split()) for text in texts),
        "rates": {word: round(rate(texts, word), 2) for word in ["the", "a", "is", "you", "i", "listen", "yes", "good"]},
    }
json.dump(summary, open(os.path.join(WORK_DIR, "data", "surface_metrics.json"), "w"), indent=1)

header = f"{'set':<16}{'cos':>6}{'dist':>6}{'openL':>7}{'LL':>6}{'GGG':>6}{'tag?':>6}{'words':>7}  rates the/a/you/i/listen/yes/good"
print(header)
for name, row in summary.items():
    rates = row["rates"]
    print(f"{name:<16}{row['style_cosine']:>6.3f}{row['centroid_distance']:>6.2f}{row['opens_listen']:>7.0%}{row['has_listen_listen']:>6.0%}{row['closes_good_good']:>6.0%}{row['question_tag']:>6.0%}{row['mean_words']:>7.1f}  {rates['the']}/{rates['a']}/{rates['you']}/{rates['i']}/{rates['listen']}/{rates['yes']}/{rates['good']}")

identical = sum(left == right for left, right in zip(sets["SFT (gen A)"], sets["SFT (gen B)"]))
print(f"\nSFT gen A vs gen B identical outputs: {identical}/200")
