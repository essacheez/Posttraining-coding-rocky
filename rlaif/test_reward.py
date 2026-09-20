"""Offline checks for the GRPO reward. No network, no API key, no GPU.

    uv run python rlaif/test_reward.py

The logic under test is small but load-bearing: it decides what an unscorable or
truncated rollout is worth, and getting that wrong biases every group it appears
in rather than failing loudly.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JUDGE_SCORES_LOG", os.devnull)

from rlaif import reward as rm

PASS, FAIL = [], []


def check(name, condition):
    (PASS if condition else FAIL).append(name)
    print(f"  {'ok ' if condition else 'FAIL'} {name}")


def scored(structure, vocab, markers, consistency, human):
    return {
        "sentence_structure": structure,
        "vocabulary": vocab,
        "markers": markers,
        "consistency": consistency,
        "human_plausibility": human,
    }


def run(completions, scores, lengths=None):
    """Drive persona_reward with a canned judge result, no network."""
    original = rm.score_responses
    rm.score_responses = lambda texts, complete, **kwargs: scores
    rm._complete = lambda prompt: ""
    ids = [[0] * (length or 1) for length in (lengths or [1] * len(completions))]
    try:
        return rm.persona_reward(
            prompts=[None] * len(completions), completions=completions, completion_ids=ids
        )
    finally:
        rm.score_responses = original


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

print("\ncompletion text")
check("a conversational completion is unwrapped", rm.completion_text([{"role": "a", "content": "hi"}]) == "hi")
check("a plain string passes through", rm.completion_text("hi") == "hi")

print("\nunscorable rollouts")
rewards = run(["a", "b", "c"], [scored(5, 5, 5, 5, 5), None, scored(1, 1, 1, 1, 1)])
check("one reward per completion, order preserved", len(rewards) == 3)
check("an unscored rollout is None, for TRL to exclude", rewards[1] is None)
check("a scored rollout is untouched", rewards[0] == 25 and rewards[2] == 5)

print("\ntruncation")
rewards = run(
    ["a", "b"],
    [scored(5, 5, 5, 5, 5), scored(5, 5, 5, 5, 5)],
    lengths=[10, rm.MAX_COMPLETION_LENGTH],
)
check("a completion at the length cap is not scored on its merits", rewards[1] is None)
check("one below the cap is", rewards[0] == 25)

print("\nwhole batch unusable")
check("no usable score is all-None, not a crash", run(["a", "b"], [None, None]) == [None, None])

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
