"""Figure 2: marker fingerprint, each rate divided by the human reference rate, log scale."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from style import *

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
setup()
surface = json.load(open(os.path.join(WORK, "data", "surface_metrics.json")))
reference = surface["human reference"]
REPO = os.path.dirname(WORK)
# base Qwen generations are not committed, only its word rates from the SFT persona eval
base_rates = json.load(open(os.path.join(REPO, "EVALS", "results_sft", "persona_eval_results_sft.json")))["base"]["marker_rates_per_1k_tokens"]
FLOOR = 0.01

features = [
    ("listen", lambda row: row["rates"]["listen"], '"listen" /1k'),
    ("yes", lambda row: row["rates"]["yes"], '"yes" /1k'),
    ("you", lambda row: row["rates"]["you"], '"you" /1k'),
    ("i", lambda row: row["rates"]["i"], '"I" /1k'),
    ("good", lambda row: row["rates"]["good"], '"good" /1k'),
    ("tag", lambda row: row["question_tag"], '", yes?" / ", question?" tag'),
    ("closer", lambda row: row["closes_good_good"], 'ends "Good, good, good"'),
    ("the", lambda row: row["rates"]["the"], '"the" /1k'),
]
series = [("SFT (gen B)", "SFT", BLUE, "o"), ("v1 step 112", "first RLAIF attempt (abandoned)", ORANGE, "s"), ("v2 final", "RLAIF (final)", AQUA, "D")]

fig, ax = plt.subplots(figsize=(8.5, 5.0))
positions = np.arange(len(features))[::-1]
for offset, (key, label, color, marker) in zip([-0.24, -0.06, 0.12], series):
    ratios = [max(func(surface[key]) / func(reference), FLOOR) for _, func, _ in features]
    ax.scatter(ratios, positions + offset, color=color, marker=marker, s=55, edgecolor=SURFACE, linewidth=1.5, label=label, zorder=3)
base_points = []
for position, (name, func, _) in zip(positions, features):
    if name in base_rates:
        base_points.append((max(base_rates[name] / func(reference), FLOOR), position + 0.36))
ax.scatter(*zip(*base_points), facecolor="none", edgecolor=INK2, marker="o", s=50, linewidth=1.3, label="base Qwen (word rates only)", zorder=3)
ax.axvline(1, color=INK2, linewidth=1.2)
ax.axvspan(0.5, 2, color=GRID, alpha=0.5, zorder=0)
ax.set_xscale("log")
ax.set_xticks([0.01, 0.03, 0.1, 0.25, 0.5, 1, 2, 4, 8, 25])
ax.set_xticklabels(["<0.01x", "0.03x", "0.1x", "0.25x", "0.5x", "1x\n(reference)", "2x", "4x", "8x", "25x"])
ax.set_yticks(positions)
ax.set_yticklabels([label for _, _, label in features])
ax.grid(axis="y", visible=False)
ax.set_xlabel("rate relative to the held-out reference replies (log scale; shaded band = within 2x)")
ax.set_title("Verbal tics: SFT overuses the catchphrases, the first attempt amplifies, RLAIF overcorrects")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8.5)
fig.savefig(os.path.join(WORK, "figures", "fig2_catchphrase_rates.png"), dpi=200, bbox_inches="tight")
print("ok")
