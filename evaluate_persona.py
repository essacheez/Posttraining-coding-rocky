"""Measure how closely model outputs match the Rocky persona in the SFT dataset.

Scores each checkpoint two ways against the training corpus: sentence-transformer
cosine (semantic) and a function-word profile (stylometric). Generations are cached
to disk, so --score-only recomputes the metrics without a GPU.

    uv run python evaluate_persona.py --base
    uv run python evaluate_persona.py --base --checkpoints outputs/sft/final_adapter
    uv run python evaluate_persona.py --score-only
"""

import argparse
import json
import math
import os
import re
import statistics
import time
from collections import Counter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
# mirrors train_sft.OUTPUT_DIR, local so --score-only runs without the training deps
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "sft")
TRAIN_FILE = os.path.join(PROJECT_DIR, "data", "raw", "train_rocky.jsonl")
TEST_FILE = os.path.join(PROJECT_DIR, "data", "raw", "test_rocky.jsonl")
GENERATIONS_FILE = os.path.join(OUTPUT_DIR, "persona_generations.json")
RESULTS_FILE = os.path.join(OUTPUT_DIR, "persona_eval_results.json")

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
MAX_NEW_TOKENS = 512

# function words carry voice while staying near-independent of topic
FUNCTION_WORDS = [
    "a", "about", "after", "all", "also", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "but", "by", "can", "could",
    "did", "do", "does", "for", "from", "had", "has", "have", "he", "her",
    "here", "him", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "just", "may", "me", "might", "more", "most", "much", "must", "my", "no",
    "not", "now", "of", "on", "one", "only", "or", "other", "our", "out",
    "over", "own", "said", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "up", "us", "very", "was", "we",
    "were", "what", "when", "where", "which", "while", "who", "will", "with",
    "would", "you", "your",
]
PUNCTUATION = [".", ",", "!", "?", ";", ":", '"', "'", "-", "(", ")"]

# Rocky drops articles and copulas and leans on a few discourse markers, the most
# legible evidence of transfer
MARKER_WORDS = ["the", "a", "is", "an", "was", "i", "good", "listen", "yes"]


def read_jsonl(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as jsonl_file:
        return [json.loads(raw_line) for raw_line in jsonl_file if raw_line.strip()]


def record_messages(record):
    """raw/*_rocky.jsonl stores messages, persona_*_data.jsonl the prepared
    prompt/completion form of the same examples."""
    if "messages" in record:
        return record["messages"]
    return record["prompt"] + record["completion"]


def load_eval_examples(num_problems):
    examples = []
    for record in read_jsonl(TEST_FILE):
        messages = record_messages(record)
        assert messages[-1]["role"] == "assistant", "last turn must be an assistant message"
        examples.append({"prompt": messages[:-1], "reference": messages[-1]["content"]})
    return examples[:num_problems] if num_problems else examples


def load_persona_corpus():
    """Every assistant turn in the training set, i.e. the target style."""
    return [
        message["content"]
        for record in read_jsonl(TRAIN_FILE)
        for message in record_messages(record)
        if message["role"] == "assistant"
    ]


def build_eval_prompt(tokenizer, prompt_messages):
    """Same system prompt as training, so the persona has to come from weights."""
    from evaluation import SYSTEM_PROMPT, build_prompt

    if len(prompt_messages) == 1:
        return build_prompt(tokenizer, prompt_messages[0]["content"])
    messages = list(prompt_messages)
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_all(checkpoints, examples):
    from evaluation import lora_request_for
    from train_sft import LORA_RANK, MODEL_ID
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(model=MODEL_ID, enable_lora=True, max_lora_rank=LORA_RANK, dtype="bfloat16")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    prompts = [build_eval_prompt(tokenizer, example["prompt"]) for example in examples]

    generations = {}
    for checkpoint_name, adapter_path in checkpoints:
        lora_request = lora_request_for(adapter_path)
        generation_start = time.time()
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        generations[checkpoint_name] = [output.outputs[0].text.strip() for output in outputs]
        print(
            f"{checkpoint_name}: generated {len(prompts)} completions "
            f"in {time.time() - generation_start:.1f}s"
        )
    return generations


def tokenize(text):
    return re.findall(r"[a-z']+", text.lower())


def sentence_lengths(text):
    sentences = [part for part in re.split(r"[.!?]+", text) if part.strip()]
    return [len(part.split()) for part in sentences] or [0]


def style_features(text):
    tokens = tokenize(text)
    token_count = max(len(tokens), 1)
    counts = Counter(tokens)
    lengths = sentence_lengths(text)

    features = {
        f"word:{word}": 1000 * counts[word] / token_count for word in FUNCTION_WORDS
    }
    features.update(
        {
            f"punct:{mark}": 1000 * text.count(mark) / max(len(text), 1)
            for mark in PUNCTUATION
        }
    )
    features["mean_sentence_length"] = statistics.mean(lengths)
    features["std_sentence_length"] = statistics.pstdev(lengths)
    features["mean_word_length"] = sum(len(token) for token in tokens) / token_count
    features["type_token_ratio"] = len(set(tokens)) / token_count
    features["uppercase_ratio"] = sum(c.isupper() for c in text) / max(len(text), 1)
    return features


FEATURE_NAMES = sorted(style_features("placeholder text."))


def feature_vector(text):
    features = style_features(text)
    return [features[name] for name in FEATURE_NAMES]


def cosine(left, right):
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def profile_vector(texts):
    """Profile of the concatenated corpus, so short responses do not weigh as much
    as long ones."""
    return feature_vector("\n\n".join(texts))


def standardizer(texts):
    vectors = [feature_vector(text) for text in texts]
    means = [statistics.mean(column) for column in zip(*vectors)]
    stds = [statistics.pstdev(column) or 1.0 for column in zip(*vectors)]
    return means, stds


def centroid_distance(texts, means, stds):
    """Mean distance to the persona centroid in std-units, so ~1.0 is about as close
    as a typical training example."""
    distances = []
    for text in texts:
        vector = feature_vector(text)
        standardized = [(value - mean) / std for value, mean, std in zip(vector, means, stds)]
        distances.append(math.sqrt(sum(z * z for z in standardized) / len(standardized)))
    return statistics.mean(distances)


def marker_rates(texts):
    tokens = tokenize("\n\n".join(texts))
    token_count = max(len(tokens), 1)
    counts = Counter(tokens)
    return {word: round(1000 * counts[word] / token_count, 2) for word in MARKER_WORDS}


def embed_texts(model, texts, batch_size):
    return model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True
    )


def embedding_scores(generations, references, persona_corpus, device, batch_size):
    from sentence_transformers import SentenceTransformer

    # CPU by default, vLLM preallocates most of the GPU and co-locating tends to OOM
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    corpus_embeddings = embed_texts(model, persona_corpus, batch_size)
    corpus_centroid = corpus_embeddings.mean(axis=0)
    corpus_centroid = corpus_centroid / max(float((corpus_centroid**2).sum() ** 0.5), 1e-12)
    reference_embeddings = embed_texts(model, references, batch_size)

    scores = {}
    for checkpoint_name, texts in generations.items():
        generated = embed_texts(model, texts, batch_size)
        paired = float((generated * reference_embeddings).sum(axis=1).mean())
        to_centroid = float((generated @ corpus_centroid).mean())
        scores[checkpoint_name] = {
            "embed_cosine_paired": paired,
            "embed_cosine_centroid": to_centroid,
        }
    return scores


def score_generations(generations, examples, persona_corpus, args):
    references = [example["reference"] for example in examples]
    for checkpoint_name, texts in generations.items():
        if len(texts) != len(references):
            raise SystemExit(
                f"{checkpoint_name} has {len(texts)} generations but {len(references)} "
                f"eval prompts, rerun without --num-problems or regenerate"
            )

    persona_profile = profile_vector(persona_corpus)
    means, stds = standardizer(persona_corpus)

    results = {}
    for checkpoint_name, texts in generations.items():
        results[checkpoint_name] = {
            "style_cosine": cosine(profile_vector(texts), persona_profile),
            "style_centroid_distance": centroid_distance(texts, means, stds),
            "marker_rates_per_1k_tokens": marker_rates(texts),
            "mean_response_words": statistics.mean(len(text.split()) for text in texts),
        }

    # the held-out references score what a perfect persona match looks like
    results["reference (ceiling)"] = {
        "style_cosine": cosine(profile_vector(references), persona_profile),
        "style_centroid_distance": centroid_distance(references, means, stds),
        "marker_rates_per_1k_tokens": marker_rates(references),
        "mean_response_words": statistics.mean(len(text.split()) for text in references),
    }

    if not args.no_embeddings:
        embedding_targets = dict(generations)
        embedding_targets["reference (ceiling)"] = references
        for checkpoint_name, scores in embedding_scores(
            embedding_targets, references, persona_corpus, args.device, args.batch_size
        ).items():
            results[checkpoint_name].update(scores)
    return results


def print_table(results):
    columns = [
        ("style_cosine", "style cos", "higher"),
        ("style_centroid_distance", "style dist", "lower"),
        ("embed_cosine_paired", "embed paired", "higher"),
        ("embed_cosine_centroid", "embed centroid", "higher"),
    ]
    columns = [column for column in columns if any(column[0] in row for row in results.values())]

    name_width = max(len(name) for name in results)
    header = f"{'checkpoint':<{name_width}}  " + "  ".join(
        f"{label:>14}" for _, label, _ in columns
    )
    print("\n" + header)
    print("-" * len(header))
    for checkpoint_name, scores in results.items():
        cells = "  ".join(
            f"{scores[key]:>14.3f}" if key in scores else f"{'-':>14}"
            for key, _, _ in columns
        )
        print(f"{checkpoint_name:<{name_width}}  {cells}")
    print("\n" + "  ".join(f"{label} ({direction} is closer)" for _, label, direction in columns))

    print("\nmarker rates per 1k tokens (Rocky drops articles/copulas):")
    marker_header = f"{'checkpoint':<{name_width}}  " + "  ".join(
        f"{word:>7}" for word in MARKER_WORDS
    )
    print(marker_header)
    print("-" * len(marker_header))
    for checkpoint_name, scores in results.items():
        rates = scores["marker_rates_per_1k_tokens"]
        cells = "  ".join(f"{rates[word]:>7.2f}" for word in MARKER_WORDS)
        print(f"{checkpoint_name:<{name_width}}  {cells}")


def parse_args():
    parser = argparse.ArgumentParser(description="Measure persona-style match against the SFT dataset")
    parser.add_argument("--base", action="store_true", help="also evaluate the base model with no adapter")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="adapter dirs to evaluate, defaults to all checkpoints under outputs/sft plus final_adapter",
    )
    parser.add_argument(
        "--num-problems", type=int, default=None, help="subsample the test set, defaults to all 200"
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help=f"rescore the generations cached in {GENERATIONS_FILE} without loading a model",
    )
    parser.add_argument("--no-embeddings", action="store_true", help="stylometry only, skips sentence-transformers")
    parser.add_argument("--device", default="cpu", help="device for the embedding model, defaults to cpu")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding batch size")
    return parser.parse_args()


def main():
    args = parse_args()
    examples = load_eval_examples(args.num_problems)
    persona_corpus = load_persona_corpus()
    print(f"{len(examples)} eval prompts, {len(persona_corpus)} persona reference turns")

    if args.score_only:
        with open(GENERATIONS_FILE, encoding="utf-8") as generations_in:
            generations = json.load(generations_in)
        print(f"scoring cached generations from {GENERATIONS_FILE}")
    else:
        checkpoints = []
        if args.base:
            checkpoints.append(("base", None))
        if args.checkpoints:
            checkpoints += [(os.path.basename(path.rstrip("/")), path) for path in args.checkpoints]
        elif not args.base:
            from evaluation import find_checkpoints

            checkpoints += [(os.path.basename(path), path) for path in find_checkpoints()]
        if not checkpoints:
            raise SystemExit("nothing to evaluate, pass --base and/or --checkpoints")

        generations = generate_all(checkpoints, examples)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(GENERATIONS_FILE, "w", encoding="utf-8") as generations_out:
            json.dump(generations, generations_out, indent=2)
        print(f"wrote generations to {GENERATIONS_FILE}")

    results = score_generations(generations, examples, persona_corpus, args)
    print_table(results)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as results_out:
        json.dump(results, results_out, indent=2)
    print(f"\nwrote results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
