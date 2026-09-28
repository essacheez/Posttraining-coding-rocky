"""Figure 6: v1 reward hacking, four panels on a shared step axis."""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from style import *

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
setup()
log = json.load(open(os.path.join(WORK, "data", "first_rlaif_attempt", "trainer_state.json")))["log_history"]
train = [row for row in log if "reward" in row]
evals = [row for row in log if "eval_reward" in row]
coding = json.load(open(os.path.join(WORK, "data", "first_rlaif_attempt", "eval_results.json")))
surface = json.load(open(os.path.join(WORK, "data", "surface_metrics.json")))

steps = np.array([row["step"] for row in train])
reward = np.array([row["reward"] for row in train])
window = 7
smooth = np.convolve(reward, np.ones(window) / window, mode="valid")

ckpt_steps = [0, 15, 45, 75, 105, 112]
ckpt_keys = ["base", "checkpoint-15", "checkpoint-45", "checkpoint-75", "checkpoint-105", "checkpoint-112"]
set_keys = ["SFT (gen A)", "v1 step 15", "v1 step 45", "v1 step 75", "v1 step 105", "v1 step 112"]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), gridspec_kw={"wspace": 0.28})

ax = axes[0]
ax.plot(steps, reward, color=ORANGE, alpha=0.25, linewidth=1)
ax.plot(steps[window - 1:], smooth, color=ORANGE, label="train batch reward (7-step mean)")
ax.plot([row["step"] for row in evals], [row["eval_reward"] for row in evals], "o", color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=2, label="held-out val reward")
ax.set_title("A. Judge reward (0 to 10) rises", fontsize=10)
ax.text(78, 4.2, "log ends\nat step 75", fontsize=8, color=INK2)
ax.set_ylabel("reward")
ax.legend(loc="upper left", fontsize=8.5)

ax = axes[1]
ax.plot(ckpt_steps, [coding[key]["humaneval"] for key in ckpt_keys], "-o", color=INK2, markeredgecolor=SURFACE, markeredgewidth=2, label="HumanEval")
ax.plot(ckpt_steps, [coding[key]["mbpp"] for key in ckpt_keys], "-o", color=VIOLET, markeredgecolor=SURFACE, markeredgewidth=2, label="MBPP")
ax.set_title("B. Coding collapses after step 45", fontsize=10)
ax.set_ylabel("pass@1")
ax.set_ylim(0, 0.7)
ax.legend(loc="lower left", fontsize=8.5)
ax.annotate("first-fence-only\nextractor (Sec. 7)", (75, 0.067), xytext=(62, 0.3), fontsize=8, color=INK2, arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8})

ax = axes[2]
ax.plot(ckpt_steps, [100 * surface[key]["opens_listen"] for key in set_keys], "-o", color=ORANGE, markeredgecolor=SURFACE, markeredgewidth=2)
ax.axhline(100 * surface["human reference"]["opens_listen"], color=MUTED, linewidth=1)
ax.text(22, 100 * surface["human reference"]["opens_listen"] - 9, "reference replies: 12%", fontsize=8, color=INK2)
ax.set_title('C. Replies opening with "Listen" (%)', fontsize=10)
ax.set_ylabel("% of 200 replies")
ax.set_ylim(0, 100)

for axis in axes:
    axis.set_xlabel("GRPO step")
    axis.set_xlim(-4, 116)

fig.savefig(os.path.join(WORK, "figures", "fig5_first_attempt.png"), dpi=200, bbox_inches="tight")
print("ok")
