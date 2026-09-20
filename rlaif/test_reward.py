"""Offline checks for the GRPO reward. No network, no API key, no slime.

    uv run python rlaif/test_reward.py

The logic under test is small but load-bearing: it decides what an unscorable or
truncated rollout is worth, and getting that wrong biases every group it appears
in rather than failing loudly.
"""

import asyncio
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JUDGE_SCORES_LOG", os.devnull)

from rlaif import reward as rm

PASS, FAIL = [], []


def check(name, condition):
    (PASS if condition else FAIL).append(name)
    print(f"  {'ok ' if condition else 'FAIL'} {name}")


def sample(response, status="COMPLETED"):
    return types.SimpleNamespace(
        response=response, status=types.SimpleNamespace(name=status), rollout_id=0
    )


def run(samples, scores):
    """Drive batched_custom_rm with a canned judge result, no network."""
    rm._complete = lambda prompt: ""
    original = rm.score_responses
    rm.score_responses = lambda responses, complete, **kwargs: scores
    try:
        return asyncio.run(rm.batched_custom_rm(None, samples))
    finally:
        rm.score_responses = original


def scored(structure, vocab, markers, consistency, human):
    return {
        "sentence_structure": structure,
        "vocabulary": vocab,
        "markers": markers,
        "consistency": consistency,
        "human_plausibility": human,
    }


print("reward modes")
check("gated is the default", rm.REWARD_MODE == "gated")
rm.REWARD_MODE = "sum"
check("sum is the plain 5-25 total", rm.to_scalar(scored(5, 5, 5, 5, 2)) == 22)
rm.REWARD_MODE = "gated"
check("gating scales by human_plausibility", rm.to_scalar(scored(5, 5, 5, 5, 2)) == 22 * 2 / 5)
check(
    "gating ranks the in-character response above the marker-stuffed one",
    rm.to_scalar(scored(5, 5, 3, 5, 4)) > rm.to_scalar(scored(5, 5, 5, 5, 2)),
)
rm.REWARD_MODE = "nonsense"
try:
    rm.to_scalar(scored(5, 5, 5, 5, 5))
    check("an unknown mode raises", False)
except ValueError:
    check("an unknown mode raises", True)
rm.REWARD_MODE = "sum"

print("\nunscorable rollouts")
rewards = run([sample("a"), sample("b"), sample("c")], [scored(5, 5, 5, 5, 5), None, scored(1, 1, 1, 1, 1)])
check("one reward per sample, order preserved", len(rewards) == 3)
check("an unscored rollout takes the group mean", rewards[1] == (25 + 5) / 2)
check("a scored rollout is untouched", rewards[0] == 25 and rewards[2] == 5)

print("\ntruncation")
rewards = run(
    [sample("a"), sample("b", status="TRUNCATED")],
    [scored(5, 5, 5, 5, 5), scored(5, 5, 5, 5, 5)],
)
check("a truncated rollout is not scored on its own merits", rewards[1] == 25.0)
check("  (it takes the group mean, so its advantage is ~0)", rewards[1] == rewards[0])

print("\nwhole group unusable")
rewards = run([sample("a"), sample("b")], [None, None])
check("no usable score is a flat group, not a crash", rewards == [0.0, 0.0])

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
