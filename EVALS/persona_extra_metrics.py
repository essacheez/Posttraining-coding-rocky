import argparse
import json
import math
import os
import re
import sys

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EVALS_DIR)

from evaluate_persona import feature_vector, load_eval_examples, load_persona_corpus, standardizer

GAMING_WORDS = ["you", "listen", "yes", "good"]
# a response repeating "listen" this often is a greedy decoding loop, not style
DEGENERATE_LISTEN_COUNT = 20


def word_rate_per_1k(texts, word):
    total = 0
    hits = 0
    for text in texts:
        words = re.findall(r"[a-z']+", text.lower())
        total += len(words)
        hits += words.count(word)
    return 1000.0 * hits / max(total, 1)


def offset_and_spread(texts, means, stds):
    """Offset is the distance of the generation mean from the persona mean in z units, spread the mean distance to that generation mean."""
    vectors = []
    for text in texts:
        raw = feature_vector(text)
        vectors.append([(value - mean) / std for value, mean, std in zip(raw, means, stds)])
    dimension = len(vectors[0])
    center = [sum(vector[index] for vector in vectors) / len(vectors) for index in range(dimension)]
    offset = math.sqrt(sum(value * value for value in center) / dimension)
    spread = 0.0
    for vector in vectors:
        spread += math.sqrt(sum((value - c) ** 2 for value, c in zip(vector, center)) / dimension)
    return offset, spread / len(vectors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True)
    args = parser.parse_args()

    generations = json.load(open(args.generations, encoding="utf-8"))
    examples = load_eval_examples(None)
    references = [example["reference"] for example in examples]
    persona_corpus = load_persona_corpus()
    means, stds = standardizer(persona_corpus)

    sets = dict(generations)
    sets["reference"] = references
    # word rates exclude degenerate listen loops; offset, spread and mean words keep every response
    header = f"{'checkpoint':<16} {'offset':>7} {'spread':>7} {'you/1k':>7} {'listen/1k':>10} {'yes/1k':>7} {'good/1k':>8} {'starts Listen':>14} {'mean words':>11} {'loops':>6}"
    print(header)
    print("-" * len(header))
    for name, texts in sets.items():
        offset, spread = offset_and_spread(texts, means, stds)
        clean = [text for text in texts if len(re.findall(r"(?i)\blisten\b", text)) < DEGENERATE_LISTEN_COUNT]
        rates = [word_rate_per_1k(clean, word) for word in GAMING_WORDS]
        starts_listen = sum(1 for text in texts if text.lstrip().lower().startswith("listen"))
        mean_words = sum(len(text.split()) for text in texts) / len(texts)
        degenerate = len(texts) - len(clean)
        print(f"{name:<16} {offset:>7.3f} {spread:>7.3f} {rates[0]:>7.1f} {rates[1]:>10.1f} {rates[2]:>7.1f} {rates[3]:>8.1f} {starts_listen:>10}/{len(texts):<3} {mean_words:>11.1f} {degenerate:>6}")


if __name__ == "__main__":
    main()
