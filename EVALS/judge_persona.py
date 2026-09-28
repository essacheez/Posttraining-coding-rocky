"""Persona adherence with the five dimension rubric judge, scored on cached generations.
Reads generations written by evaluate_persona.py and never regenerates, so no GPU is needed."""

import argparse
import json
import os
import sys

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVALS_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, EVALS_DIR)

from evaluate_persona import load_eval_examples
from judge.scorer import DIMENSIONS, JUDGE_MODEL, JUDGE_WORKERS, aggregate, score_responses

DEFAULT_GENERATIONS = os.path.join(EVALS_DIR, "results_rlaif", "persona_generations.json")
DEFAULT_RESULTS_DIR = os.path.join(EVALS_DIR, "results_judge")
REFERENCE_NAME = "reference (ceiling)"

TABLE_COLUMNS = [
    ("judge_sentence_structure", "sentence"),
    ("judge_vocabulary", "vocab"),
    ("judge_markers", "markers"),
    ("judge_consistency", "consist"),
    ("judge_human_plausibility", "human"),
    ("judge_total", "total"),
    ("judge_gaming_gap", "gap"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Score cached persona generations with the rubric judge")
    parser.add_argument("--generations", default=DEFAULT_GENERATIONS, help="persona_generations.json to score")
    parser.add_argument("--stages", nargs="+", default=None, help="which generation keys to score, defaults to all")
    parser.add_argument("--no-reference", action="store_true", help="skip scoring the held-out human references")
    parser.add_argument("--num-problems", type=int, default=None, help="score only the first N prompts, for a cheap trial run")
    parser.add_argument("--model", default=JUDGE_MODEL, help="judge model")
    parser.add_argument("--workers", type=int, default=JUDGE_WORKERS, help="concurrent judge calls")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="where judge_scores.json and judge_results.json go")
    return parser.parse_args()


def print_table(results):
    name_width = max(len(name) for name in results)
    header = f"{'stage':<{name_width}}  " + "  ".join(f"{label:>8}" for key, label in TABLE_COLUMNS) + "  scored"
    print("\n" + header)
    print("-" * len(header))
    for name, summary in results.items():
        cells = []
        for key, label in TABLE_COLUMNS:
            cells.append(f"{summary[key]:>8.2f}" if key in summary else f"{'-':>8}")
        scored = f"{summary['judge_scored']}/{summary['judge_scored'] + summary['judge_unscored']}"
        print(f"{name:<{name_width}}  " + "  ".join(cells) + f"  {scored:>6}")
    print("\ndimensions are 1 to 5 and total is 5 to 25, higher is better.")
    print("gap is markers minus human plausibility; a gap that grows across stages means the judge is being gamed.")


def main():
    args = parse_args()
    with open(args.generations, encoding="utf-8") as generations_file:
        generations = json.load(generations_file)

    stage_names = args.stages or list(generations)
    targets = {}
    for stage_name in stage_names:
        targets[stage_name] = generations[stage_name][: args.num_problems]
    if not args.no_reference:
        examples = load_eval_examples(args.num_problems)
        targets[REFERENCE_NAME] = [example["reference"] for example in examples]

    print(f"judging {len(targets)} stages x {len(next(iter(targets.values())))} responses with {args.model}")
    per_response = {}
    results = {}
    for stage_name, response_texts in targets.items():
        print(f"scoring {stage_name}")
        scored = score_responses(response_texts, model=args.model, workers=args.workers)
        per_response[stage_name] = scored
        results[stage_name] = aggregate(scored)

    print_table(results)

    os.makedirs(args.results_dir, exist_ok=True)
    scores_path = os.path.join(args.results_dir, "judge_scores.json")
    results_path = os.path.join(args.results_dir, "judge_results.json")
    run_info = {"judge_model": args.model, "generations": os.path.relpath(args.generations, REPO_ROOT), "dimensions": DIMENSIONS}
    with open(scores_path, "w", encoding="utf-8") as scores_file:
        json.dump({"run": run_info, "scores": per_response}, scores_file, indent=2)
    with open(results_path, "w", encoding="utf-8") as results_file:
        json.dump({"run": run_info, "results": results}, results_file, indent=2)
    print(f"\nwrote {scores_path}\nwrote {results_path}")


if __name__ == "__main__":
    main()
