"""Draw the two RLVR figures used in the main README: coding scores across training steps,
and the curriculum (problems available vs picked per difficulty band)."""

import json
import os

import matplotlib.pyplot as plt

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(FIGURES_DIR))
RUN_RESULTS_FILE = os.path.join(REPO_ROOT, "EVALS", "results_rlvr", "eval_results.json")
BASE_RESULTS_FILE = os.path.join(REPO_ROOT, "EVALS", "results_base", "eval_results.json")
CODING_FIGURE = os.path.join(FIGURES_DIR, "rlvr_coding_by_step.png")
CURRICULUM_FIGURE = os.path.join(FIGURES_DIR, "rlvr_curriculum.png")

# step 0 is the RLAIF model RLVR starts from
RLAIF_SCORES = {"humaneval": 0.579, "humanevalplus": 0.537, "mbpp": 0.562}
CHECKPOINT_STEPS = {
    "checkpoint-25": 25,
    "checkpoint-50": 50,
    "checkpoint-75": 75,
    "checkpoint-100": 100,
    "checkpoint-125": 125,
    "checkpoint-150": 150,
    "RLVR_model": 174,
}
BENCHMARK_TITLES = {"humaneval": "HumanEval", "humanevalplus": "HumanEval+", "mbpp": "MBPP"}

# difficulty band, passing rollouts out of 8, problems available in the pool, problems picked
CURRICULUM_BANDS = [
    ("too easy", "7-8", 38, 0),
    ("easy", "5-6", 113, 113),
    ("mid", "3-4", 201, 201),
    ("hard", "1-2", 318, 316),
    ("very hard", "0", 1030, 70),
]

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES_BLUE = "#2a78d6"
AVAILABLE_BLUE = "#86b6ef"
PICKED_BLUE = "#1c5cab"
REFERENCE_GRAY = "#52514e"


def read_json(json_path):
    with open(json_path, encoding="utf-8") as json_file:
        return json.load(json_file)


def style_axis(axis):
    axis.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        axis.spines[side].set_visible(False)
    axis.spines["bottom"].set_color(GRID)
    axis.tick_params(colors=TEXT_SECONDARY, length=0, labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def plot_coding_by_step():
    run_results = read_json(RUN_RESULTS_FILE)
    base_results = read_json(BASE_RESULTS_FILE)["base"]
    steps = [0] + list(CHECKPOINT_STEPS.values())

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True, facecolor=SURFACE)
    for axis, (benchmark, title) in zip(axes, BENCHMARK_TITLES.items()):
        scores = [RLAIF_SCORES[benchmark]] + [run_results[name][benchmark] for name in CHECKPOINT_STEPS]
        base_score = base_results[benchmark]
        style_axis(axis)

        axis.axhline(base_score, color=REFERENCE_GRAY, linewidth=1.2, linestyle=(0, (4, 3)))
        axis.text(2, base_score + 0.006, f"base model {base_score:.3f}", color=TEXT_SECONDARY, fontsize=8.5)
        axis.plot(steps, scores, color=SERIES_BLUE, linewidth=2, marker="o", markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.5)

        # label only the start and the end, the numbers the text talks about
        axis.annotate(f"RLAIF {scores[0]:.3f}", (steps[0], scores[0]), xytext=(8, -4),
                      textcoords="offset points", color=TEXT_SECONDARY, fontsize=8.5, va="top")
        # the MBPP end point sits just under the base line, so its label goes below the point
        end_offset = (-12, -16) if scores[-1] < base_score and base_score - scores[-1] < 0.02 else (-12, 8)
        axis.annotate(f"{scores[-1]:.3f}", (steps[-1], scores[-1]), xytext=end_offset,
                      textcoords="offset points", color=TEXT_PRIMARY, fontsize=9, fontweight="bold")

        axis.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", fontweight="bold")
        axis.set_xticks([0, 25, 50, 75, 100, 125, 150, 174])
        axis.set_xlabel("training step", color=TEXT_SECONDARY, fontsize=9)

    axes[0].set_ylabel("pass@1", color=TEXT_SECONDARY, fontsize=9)
    axes[0].set_ylim(0.50, 0.79)
    figure.suptitle("Coding pass@1 across RLVR training (step 0 is the RLAIF model)",
                    color=TEXT_PRIMARY, fontsize=12, x=0.01, ha="left")
    figure.tight_layout()
    figure.savefig(CODING_FIGURE, dpi=160, facecolor=SURFACE)
    print(f"wrote {CODING_FIGURE}")


def plot_curriculum():
    labels = [f"{band}  ({passes}/8 pass)" for band, passes, available, picked in CURRICULUM_BANDS]
    available_counts = [available for band, passes, available, picked in CURRICULUM_BANDS]
    picked_counts = [picked for band, passes, available, picked in CURRICULUM_BANDS]
    positions = list(range(len(labels)))[::-1]

    figure, axis = plt.subplots(figsize=(9, 3.4), facecolor=SURFACE)
    style_axis(axis)
    axis.grid(axis="y", visible=False)
    axis.grid(axis="x", color=GRID, linewidth=0.8)

    axis.barh(positions, available_counts, height=0.62, color=AVAILABLE_BLUE, label="in the 1700 pool")
    axis.barh(positions, picked_counts, height=0.62, color=PICKED_BLUE, label="picked for training (700)")
    for position, available, picked in zip(positions, available_counts, picked_counts):
        axis.text(available + 12, position, f"{picked} of {available}", va="center",
                  color=TEXT_PRIMARY, fontsize=9)

    axis.set_yticks(positions)
    axis.set_yticklabels(labels, color=TEXT_PRIMARY, fontsize=9.5)
    axis.set_xlim(0, 1180)
    axis.set_xlabel("problems", color=TEXT_SECONDARY, fontsize=9)
    axis.legend(loc="upper right", frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    axis.set_title("Curriculum: difficulty from 8 rollouts of the RLAIF model",
                   color=TEXT_PRIMARY, fontsize=12, loc="left")
    figure.tight_layout()
    figure.savefig(CURRICULUM_FIGURE, dpi=160, facecolor=SURFACE)
    print(f"wrote {CURRICULUM_FIGURE}")


if __name__ == "__main__":
    plot_coding_by_step()
    plot_curriculum()
