"""Table and plot of STEM capability against persona adherence for every stage and attempt in stages.json.
STEM is pass@1 from the rule based verifier, persona is the rubric judge total; stages with no results yet are skipped."""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(EVALS_DIR)
DEFAULT_MANIFEST = os.path.join(EVALS_DIR, "stages.json")
DEFAULT_OUTPUT_DIR = os.path.join(EVALS_DIR, "results_rlvr")
REFERENCE_NAME = "reference (ceiling)"
VERIFIER_FILE = "verifier_results.json"

VERIFIER_COLUMNS = [("verifier_pass_at_1", "verifier pass@1"), ("verifier_assert_fraction", "assert frac"), ("hack_flagged", "hack flags")]
BENCHMARK_COLUMNS = [("humaneval", "HumanEval"), ("humanevalplus", "HumanEval+"), ("mbpp", "MBPP"), ("livecodebench", "LiveCodeBench")]
JUDGE_COLUMNS = [("judge_total", "persona (5 to 25)"), ("judge_human_plausibility", "plausibility"), ("judge_gaming_gap", "gaming gap")]
STEM_LABELS = {"verifier_pass_at_1": "held out MBPP test", "humaneval": "HumanEval", "humanevalplus": "HumanEval+", "mbpp": "MBPP test", "livecodebench": "LiveCodeBench"}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
STAGE_COLOR = "#2a78d6"
ATTEMPT_COLOR = "#eb6834"
ABLATION_COLOR = "#7a4fbf"
MARKER_SIZE = 90


def load_json(path):
    with open(path, encoding="utf-8") as json_file:
        return json.load(json_file)


def load_stage(entry):
    """Merge one stage's benchmark scores and judge summary, or return None if either is missing."""
    results_dir = os.path.join(REPO_ROOT, entry["results"])
    coding_path = os.path.join(results_dir, "eval_results.json")
    judge_path = os.path.join(results_dir, "judge_results.json")
    if not os.path.exists(coding_path) or not os.path.exists(judge_path):
        print(f"skipping {entry['name']}: no results in {entry['results']} yet")
        return None
    judge_results = load_json(judge_path)["results"]
    row = dict(entry)
    row.update(load_json(coding_path)[entry["key"]])
    row.update(judge_results[entry["key"]])
    # the verifier score is the headline STEM number, written separately by RLVR/eval_verifier.py
    verifier_path = os.path.join(results_dir, VERIFIER_FILE)
    if os.path.exists(verifier_path):
        row.update(load_json(verifier_path)[entry["key"]])
    row["reference_total"] = judge_results.get(REFERENCE_NAME, {}).get("judge_total")
    return row


def format_cell(row, key):
    if key not in row:
        return "-"
    value = row[key]
    return str(value) if isinstance(value, int) else f"{value:.3f}"


def markdown_table(rows):
    columns = VERIFIER_COLUMNS + BENCHMARK_COLUMNS + JUDGE_COLUMNS
    lines = ["| stage | " + " | ".join(label for key, label in columns) + " |"]
    lines.append("| --- |" + " --- |" * len(columns))
    for row in rows:
        name = row["name"] + (" (final)" if row.get("final") else "")
        lines.append(f"| {name} | " + " | ".join(format_cell(row, key) for key, label in columns) + " |")
    return "\n".join(lines)


def draw_arrow(axis, start, end, color):
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2, "shrinkA": 9, "shrinkB": 9, "mutation_scale": 14},
        zorder=2,
    )


def plot(rows, stem_metric, output_path):
    plt.rcParams["font.size"] = 10
    figure, axis = plt.subplots(figsize=(8, 5.5), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    plotted = [row for row in rows if stem_metric in row]
    points = {row["name"]: (row[stem_metric], row["judge_total"]) for row in plotted}
    stage_rows = [row for row in plotted if row["track"] == "stage"]
    attempt_rows = [row for row in plotted if row["track"] == "attempt"]
    ablation_rows = [row for row in plotted if row["track"] == "ablation"]

    for previous_row, next_row in zip(stage_rows, stage_rows[1:]):
        draw_arrow(axis, points[previous_row["name"]], points[next_row["name"]], STAGE_COLOR)
    # a branch row points back at the stage it forked from
    for branch_rows, color in [(attempt_rows, ATTEMPT_COLOR), (ablation_rows, ABLATION_COLOR)]:
        for row in branch_rows:
            if row.get("from") in points:
                draw_arrow(axis, points[row["from"]], points[row["name"]], color)

    tracks = [
        (stage_rows, STAGE_COLOR, "training stage"),
        (attempt_rows, ATTEMPT_COLOR, "RLVR attempt"),
        (ablation_rows, ABLATION_COLOR, "coding data ablation"),
    ]
    for track_rows, color, label in tracks:
        if not track_rows:
            continue
        x_values = [points[row["name"]][0] for row in track_rows]
        y_values = [points[row["name"]][1] for row in track_rows]
        axis.scatter(x_values, y_values, s=MARKER_SIZE, color=color, edgecolors=SURFACE, linewidths=2, zorder=3, label=label)
        for row in track_rows:
            label = row["name"] + (" (final)" if row.get("final") else "")
            axis.annotate(label, points[row["name"]], textcoords="offset points", xytext=(9, 7), color=INK_PRIMARY, fontsize=9, zorder=4)

    reference_totals = [row["reference_total"] for row in rows if row.get("reference_total")]
    if reference_totals:
        axis.axhline(reference_totals[0], color=INK_SECONDARY, linestyle=(0, (4, 4)), linewidth=1, zorder=1)
        axis.annotate(
            "human reference", (axis.get_xlim()[1], reference_totals[0]), textcoords="offset points",
            xytext=(-4, 4), color=INK_SECONDARY, fontsize=8, ha="right",
        )

    axis.set_xlabel(f"STEM capability: {STEM_LABELS.get(stem_metric, stem_metric)} pass@1 (rule based verifier)", color=INK_SECONDARY)
    axis.set_ylabel("Persona adherence: rubric judge total (5 to 25)", color=INK_SECONDARY)
    axis.set_title("Persona adherence vs STEM capability across training", color=INK_PRIMARY, loc="left", fontsize=12, pad=12)
    # room on the right for direct labels
    axis.margins(x=0.18, y=0.08)
    axis.grid(color=GRID, linewidth=0.8, zorder=0)
    axis.tick_params(colors=INK_SECONDARY, length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    legend = axis.legend(loc="lower left", frameon=False)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description="Plot persona adherence against STEM capability across stages")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="stages.json listing each stage and its results dir")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="where stages_table.md and stages_plot.png go")
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    rows = [row for row in (load_stage(entry) for entry in manifest["stages"]) if row]
    if not rows:
        raise SystemExit("no stage has both eval_results.json and judge_results.json yet")

    table = markdown_table(rows)
    print("\n" + table)
    os.makedirs(args.output_dir, exist_ok=True)
    table_path = os.path.join(args.output_dir, "stages_table.md")
    with open(table_path, "w", encoding="utf-8") as table_file:
        table_file.write(table + "\n")
    plot_path = os.path.join(args.output_dir, "stages_plot.png")
    plot(rows, manifest["stem_metric"], plot_path)
    print(f"\nwrote {table_path}\nwrote {plot_path}")


if __name__ == "__main__":
    main()
