"""Figure 1: persona (style match) against coding (HumanEval+) for every stage and attempt, from committed results."""
import json
import os
import matplotlib.pyplot as plt
from style import *

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(WORK)
setup()
sft_coding = json.load(open(os.path.join(REPO, "EVALS", "results_sft", "eval_results_sft.json")))
sft_persona = json.load(open(os.path.join(REPO, "EVALS", "results_sft", "persona_eval_results_sft.json")))
attempt_coding = json.load(open(os.path.join(WORK, "data", "first_rlaif_attempt", "eval_results.json")))
attempt_persona = json.load(open(os.path.join(WORK, "data", "first_rlaif_attempt", "persona_eval_results.json")))
rlaif_coding = json.load(open(os.path.join(REPO, "EVALS", "results_rlaif", "eval_results.json")))
rlaif_persona = json.load(open(os.path.join(REPO, "EVALS", "results_rlaif", "persona_eval_results.json")))


def point(coding, persona, key):
    return coding[key]["humanevalplus"], persona[key]["style_cosine"]


sft_final = point(rlaif_coding, rlaif_persona, "base (sft_merged)")
sft_path = [point(sft_coding, sft_persona, key) for key in ["base", "checkpoint-30", "checkpoint-70"]] + [sft_final]
attempt_path = [sft_final] + [point(attempt_coding, attempt_persona, key) for key in ["checkpoint-15", "checkpoint-45", "checkpoint-75", "checkpoint-105", "checkpoint-112"]]
rlaif_point = point(rlaif_coding, rlaif_persona, "RLAIF_model")
reference_cosine = rlaif_persona["reference (ceiling)"]["style_cosine"]

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.22})


def draw(ax, labels):
    ax.axhline(reference_cosine, color=MUTED, linewidth=1)
    ax.plot(*zip(*sft_path), "-o", color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2, label="SFT (base, step 30, step 70, final)", zorder=3)
    ax.plot(*zip(*attempt_path[:2]), "-", color=ORANGE, zorder=2)
    ax.plot(*zip(*attempt_path[1:]), "-s", color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=2, label="first RLAIF attempt (abandoned)", zorder=3)
    ax.plot([sft_final[0], rlaif_point[0]], [sft_final[1], rlaif_point[1]], "-", color=AQUA, zorder=2)
    ax.plot([rlaif_point[0]], [rlaif_point[1]], "D", color=AQUA, markeredgecolor=SURFACE, markeredgewidth=2, markersize=9, label="RLAIF (final)", zorder=4)
    for text, (xy, offset) in labels.items():
        ax.annotate(text, xy, xytext=offset, textcoords="offset points", fontsize=8.5, color=INK)


draw(ax_full, {
    "base Qwen": (sft_path[0], (-20, -15)), "SFT step 30": (sft_path[1], (8, -4)),
    "attempt, step 75": (attempt_path[3], (8, -12)), "attempt, step 112": (attempt_path[5], (8, 4)),
})
ax_full.text(0.02, reference_cosine + 0.008, "reference replies (target ceiling)", fontsize=8, color=INK2)
ax_full.add_patch(plt.Rectangle((0.50, 0.82), 0.10, 0.13, fill=False, edgecolor=INK2, linewidth=0.8, zorder=5))
ax_full.text(0.50, 0.955, "zoomed at right", fontsize=7.5, color=INK2)
ax_full.set_xlim(0, 0.75)
ax_full.set_ylim(0.55, 1.0)
ax_full.set_xlabel("coding: HumanEval+ pass@1 (164 problems, greedy)")
ax_full.set_ylabel("persona: style match to Rocky corpus")
ax_full.set_title("A. Every stage and attempt so far (RLVR pending)")
ax_full.legend(loc="lower left", fontsize=8.5)

draw(ax_zoom, {
    "SFT step 70": (sft_path[2], (-66, 2)), "SFT final": (sft_final, (8, -4)),
    "attempt, step 15": (attempt_path[1], (8, -10)), "attempt, step 45": (attempt_path[2], (-92, -4)), "RLAIF": (rlaif_point, (8, 4)),
})
ax_zoom.set_xlim(0.50, 0.60)
ax_zoom.set_ylim(0.82, 0.95)
ax_zoom.set_xlabel("HumanEval+ pass@1")
ax_zoom.set_title("B. Zoom: RLAIF vs the SFT model it started from")
fig.savefig(os.path.join(WORK, "figures", "fig1_tradeoff_all_stages.png"), dpi=200, bbox_inches="tight")
print("ok")
