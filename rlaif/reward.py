"""GRPO reward for slime: the persona judge in judge/, as a scalar.

Wired in with `--group-rm --custom-rm-path rlaif.reward.batched_custom_rm`, so
slime hands over one group at a time -- the n_samples_per_prompt completions for
a single prompt -- which is the granularity GRPO takes its advantage over.

The judge is imported rather than reimplemented: this module and
`evaluate_persona.py --judge` both go through judge.score_responses, so training
and evaluation are one instrument and the RLAIF-vs-SFT comparison holds.
"""

import asyncio
import json
import os
import statistics
import threading

from judge import DIMENSIONS, build_completer, reward, score_responses

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# gpt-4o-mini wrote the SFT rewrites and is disqualified as the judge; see
# judge/README.md on independence. scorer.py warns if pointed there anyway.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4-mini-2026-03-17")

# "gated" -- the total scaled by human_plausibility. Default because it is the
#            only dimension the SFT checkpoint scores below the human references
#            (3.38 vs 3.72) and the only one that resists marker-stuffing.
# "sum"    -- the plain 5-25 total judge/README.md defines. Kept because it is
#            the documented reward, but EVAL_RESULTS.md section 4 shows a
#            marker-stuffed response scores 22 against the SFT model's 20.59,
#            so optimizing it rewards exactly what the rubric exists to catch.
REWARD_MODE = os.environ.get("REWARD_MODE", "gated")

# Every judged rollout, for post-hoc analysis of the gaming gap over training.
# Absolute by default: Ray workers do not inherit the launch directory, so a
# relative path would scatter the log wherever each worker happens to start.
SCORES_LOG = os.environ.get(
    "JUDGE_SCORES_LOG", os.path.join(PROJECT_DIR, "outputs", "rlaif", "judge_scores.jsonl")
)

_complete = None
_log_lock = threading.Lock()


def _completer():
    global _complete
    if _complete is None:
        # score_responses does not retry, and rate limits are routine at rollout
        # concurrency; an un-retried 429 would come back as an unscored rollout
        # and quietly distort its group's advantage
        _complete = build_completer(JUDGE_MODEL, max_retries=8)
    return _complete


def to_scalar(scores):
    """The five dimensions as the single number GRPO optimizes."""
    total = float(reward(scores))
    if REWARD_MODE == "gated":
        return total * scores["human_plausibility"] / 5.0
    if REWARD_MODE == "sum":
        return total
    raise ValueError(f"unknown REWARD_MODE {REWARD_MODE!r}, expected 'gated' or 'sum'")


def _log(rollout_id, scored, truncated):
    directory = os.path.dirname(SCORES_LOG)
    with _log_lock:
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(SCORES_LOG, "a", encoding="utf-8") as log_file:
            for scores, was_truncated in zip(scored, truncated):
                if not scores:
                    continue
                row = {"rollout_id": rollout_id, "truncated": was_truncated}
                row.update({key: scores[key] for key in DIMENSIONS})
                log_file.write(json.dumps(row) + "\n")


async def batched_custom_rm(args, samples, **kwargs):
    """One group in, one reward per sample out, order preserved."""
    responses = [sample.response for sample in samples]
    scored = await asyncio.to_thread(
        score_responses, responses, _completer(), max_workers=len(responses), quiet=True
    )

    # A truncated rollout was cut off by --rollout-max-response-len, so the judge
    # marks it down on consistency for an artifact of the cap rather than for
    # anything the policy did. Still logged, but not rewarded on its own merits.
    truncated = [getattr(sample.status, "name", "") == "TRUNCATED" for sample in samples]
    _log(getattr(samples[0], "rollout_id", None), scored, truncated)

    rewards = [
        to_scalar(scores) if scores and not was_truncated else None
        for scores, was_truncated in zip(scored, truncated)
    ]

    # An unscored rollout takes its group's mean, making its advantage ~0: it
    # trains neither toward nor away from whatever it did. Flooring it instead
    # would pull the group's mean down and so distort its siblings' advantages,
    # not just its own.
    present = [value for value in rewards if value is not None]
    filler = statistics.mean(present) if present else 0.0
    return [filler if value is None else value for value in rewards]
