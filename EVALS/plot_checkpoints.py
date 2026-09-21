"""Plot the coding-vs-persona tradeoff across checkpoints: coding score
(mean of HumanEval+ and MBPP) against persona style cosine, on twin axes."""

import json
import os

import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CHECKPOINTS = ["base", "checkpoint-30", "checkpoint-70"]
LABELS = ["base", "ckpt-30", "ckpt-70"]
OUT_PATH = os.path.join(RESULTS_DIR, "coding_vs_persona_tradeoff.png")

SERIES = [
    {"label": "HumanEval+/MBPP avg", "color": "#4C72B0", "y_label": "Coding score (pass@1)", "offset": 10},
    {"label": "Persona cosine", "color": "#DD8452", "y_label": "Persona style cosine", "offset": -16},
]


def load(path):
    with open(path, encoding="utf-8") as results_file:
        return json.load(results_file)


def plot_series(ax, values, spec):
    (line,) = ax.plot(LABELS, values, marker="o", markersize=8, linewidth=2.5, color=spec["color"], label=spec["label"])
    for x, y in zip(LABELS, values):
        ax.annotate(
            f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, spec["offset"]),
            ha="center", color=spec["color"], fontsize=9, fontweight="bold",
        )
    ax.set_ylabel(spec["y_label"], color=spec["color"], fontsize=11)
    ax.tick_params(axis="y", colors=spec["color"])
    ax.spines["top"].set_visible(False)
    return line


def main():
    coding = load(os.path.join(RESULTS_DIR, "eval_results.json"))
    persona = load(os.path.join(RESULTS_DIR, "persona_eval_results.json"))
    coding_score = [(coding[c]["humanevalplus"] + coding[c]["mbpp"]) / 2 for c in CHECKPOINTS]
    style_cosine = [persona[c]["style_cosine"] for c in CHECKPOINTS]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax_coding = plt.subplots(figsize=(7, 4.5))
    ax_persona = ax_coding.twinx()
    ax_persona.grid(False)

    lines = [
        plot_series(ax_coding, coding_score, SERIES[0]),
        plot_series(ax_persona, style_cosine, SERIES[1]),
    ]

    ax_coding.set_title("Persona Tax (style vs capabilities)", fontsize=13, fontweight="bold", pad=14)
    fig.legend(handles=lines, loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    fig.savefig(OUT_PATH, dpi=200)
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
