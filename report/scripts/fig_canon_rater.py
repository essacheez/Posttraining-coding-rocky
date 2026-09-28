"""Figure 5: blind canon-grounded rating of 160 replies, three small panels sharing the source axis."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from style import *

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
setup()
summary = json.load(open(os.path.join(WORK, "data", "opus_rater_summary.json")))
sources = [("human reference", "reference\n(target)", MUTED), ("SFT (gen B)", "SFT", BLUE), ("v1 step 112", "first RLAIF\nattempt", ORANGE), ("v2 final", "RLAIF\n(final)", AQUA)]
positions = np.arange(len(sources))

fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), gridspec_kw={"wspace": 0.35})
panels = [
    ("A. Sounds like book Rocky (1 to 5)", lambda s: s["rocky_mean"], lambda s: 1.96 * s["rocky_sd"] / np.sqrt(s["n"]), (1, 5), "{:.2f}"),
    ("B. Forced catchphrases (% of replies)", lambda s: 100 * s["costume"], None, (0, 112), "{:.0f}%"),
    ("C. Fully correct (% of replies)", lambda s: 100 * s["correct_yes"], None, (0, 112), "{:.0f}%"),
]
for ax, (title, value, error, limits, fmt) in zip(axes, panels):
    for position, (key, label, color) in zip(positions, sources):
        stats = summary[key]
        ax.bar(position, value(stats) - limits[0], bottom=limits[0], width=0.6, color=color, edgecolor=SURFACE, linewidth=2)
        if error:
            ax.errorbar(position, value(stats), yerr=error(stats), color=INK2, capsize=4, linewidth=1.2)
        ax.text(position, value(stats) + (error(stats) if error else 0) + (limits[1] - limits[0]) * 0.03, fmt.format(value(stats)), ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(positions)
    ax.set_xticklabels([label for _, label, _ in sources], fontsize=8.5)
    ax.set_ylim(*limits)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", visible=False)
    if limits[1] > 100:
        ax.set_yticks([0, 25, 50, 75, 100])
axes[0].axhline(5, color=MUTED, linewidth=0.8)
fig.suptitle("Blind rating of 160 replies against a canon profile of book Rocky (Opus 5.5 raters, 40 prompts x 4 sources)", x=0.06, ha="left", fontsize=11, fontweight="bold", color=INK, y=1.03)
fig.savefig(os.path.join(WORK, "figures", "fig4_blind_canon_rating.png"), dpi=200, bbox_inches="tight")
print("ok")
