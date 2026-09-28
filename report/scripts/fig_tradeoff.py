"""Figure 1: persona/coding trade-off for the final pipeline and failed first RLAIF run."""
import json
import os

import matplotlib.pyplot as plt

from style import AQUA, BLUE, INK, INK2, MUTED, ORANGE, SURFACE, setup

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(WORK)


def load(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8") as handle:
        return json.load(handle)


setup()

base_coding = load("EVALS", "results_base", "eval_results.json")["base"]
base_persona = load("EVALS", "results_rlvr", "base", "persona_eval_results.json")["base"]
rlaif_coding = load("EVALS", "results_rlaif", "eval_results.json")
rlaif_persona = load("EVALS", "results_rlaif", "persona_eval_results.json")
rlvr_coding = load("EVALS", "results_rlvr", "eval_results.json")
rlvr_persona = load("EVALS", "results_rlvr", "persona_eval_results.json")
attempt_coding = load("report", "data", "first_rlaif_attempt", "eval_results.json")
attempt_persona = load("report", "data", "first_rlaif_attempt", "persona_eval_results.json")


def point(coding, persona, key):
    return coding[key]["humanevalplus"], persona[key]["style_cosine"]


pipeline = [
    (base_coding["humanevalplus"], base_persona["style_cosine"]),
    point(rlaif_coding, rlaif_persona, "base (sft_merged)"),
    point(rlaif_coding, rlaif_persona, "RLAIF_model"),
    point(rlvr_coding, rlvr_persona, "RLVR_model"),
]
labels = ["base Qwen", "SFT", "RLAIF", "curriculum RLVR"]

sft = pipeline[1]
failed_attempt = [sft] + [
    point(attempt_coding, attempt_persona, key)
    for key in ["checkpoint-15", "checkpoint-45", "checkpoint-75", "checkpoint-105", "checkpoint-112"]
]

fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.axhline(rlvr_persona["reference (ceiling)"]["style_cosine"], color=MUTED, linewidth=1)
ax.text(0.064, 0.986, "reference replies (style ceiling)", color=INK2, fontsize=8.5)

ax.plot(
    *zip(*failed_attempt),
    "--s",
    color=ORANGE,
    alpha=0.72,
    markeredgecolor=SURFACE,
    markeredgewidth=1.5,
    label="first RLAIF attempt (abandoned)",
    zorder=2,
)
ax.plot(
    *zip(*pipeline),
    "-o",
    color=BLUE,
    linewidth=2.8,
    markersize=8,
    markeredgecolor=SURFACE,
    markeredgewidth=1.8,
    label="final training pipeline",
    zorder=3,
)
ax.scatter([pipeline[-1][0]], [pipeline[-1][1]], marker="D", s=78, color=AQUA, edgecolor=SURFACE, linewidth=1.5, zorder=4)

offsets = [(7, -17), (7, 6), (7, 6), (7, 7)]
for label, xy, offset in zip(labels, pipeline, offsets):
    ax.annotate(label, xy, xytext=offset, textcoords="offset points", fontsize=9, color=INK)
ax.annotate("failed run, end", failed_attempt[-1], xytext=(7, 5), textcoords="offset points", fontsize=8.5, color=INK)

ax.set_xlim(0.04, 0.78)
ax.set_ylim(0.55, 1.0)
ax.set_xlabel("coding: HumanEval+ pass@1")
ax.set_ylabel("persona: style cosine to Rocky corpus")
ax.set_title("Persona and coding through the training pipeline")
ax.legend(loc="lower left", fontsize=8.5)
fig.savefig(os.path.join(WORK, "figures", "fig1_tradeoff_all_stages.png"), dpi=220, bbox_inches="tight")
print("ok")
