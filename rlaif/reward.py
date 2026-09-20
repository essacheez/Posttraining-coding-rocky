"""GRPO reward for TRL: the persona judge in judge/, as a scalar.

Passed to GRPOTrainer as `reward_funcs=persona_reward`. TRL calls it once per
generation batch with the prompts, the completions and their token ids, and
expects one reward per completion.

The judge is imported rather than reimplemented: this module and
`evaluate_persona.py --judge` both go through judge.score_responses, so training
and evaluation are one instrument and the RLAIF-vs-SFT comparison holds.
"""

import json
import os
import statistics

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

# shared with train_grpo.py's GRPOConfig so the cap the sampler enforces and the
# cap used to detect truncation here cannot drift apart
MAX_COMPLETION_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", 512))

# every judged rollout, for post-hoc analysis of the gaming gap over training
SCORES_LOG = os.environ.get(
    "JUDGE_SCORES_LOG", os.path.join(PROJECT_DIR, "outputs", "rlaif", "judge_scores.jsonl")
)

_complete = None


def _completer():
    global _complete
    if _complete is None:
        # score_responses does not retry, and rate limits are routine at rollout
        # concurrency; an un-retried 429 would come back as an unscored rollout
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


def completion_text(completion):
    """TRL hands back message dicts for a conversational prompt, a string otherwise."""
    if isinstance(completion, str):
        return completion
    return "".join(message["content"] for message in completion)


def _log(scored, truncated, step):
    directory = os.path.dirname(SCORES_LOG)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(SCORES_LOG, "a", encoding="utf-8") as log_file:
        for scores, was_truncated in zip(scored, truncated):
            if not scores:
                continue
            row = {"step": step, "truncated": was_truncated}
            row.update({key: scores[key] for key in DIMENSIONS})
            log_file.write(json.dumps(row) + "\n")


def persona_reward(prompts, completions, completion_ids, **kwargs):
    """One reward per completion. None means unscorable.

    TRL excludes a None from its group baseline and forces that completion's
    advantage to 0, so an unscorable rollout trains neither toward nor away from
    what it did, and does not distort its siblings' advantages either.
    """
    texts = [completion_text(completion) for completion in completions]
    scored = score_responses(texts, _completer(), max_workers=min(len(texts), 32), quiet=True)

    # A completion that hit the length cap was cut off mid-sentence, so the judge
    # marks it down on consistency for an artifact of the cap rather than for
    # anything the policy did. Logged, but not scored on its own merits.
    truncated = [len(ids) >= MAX_COMPLETION_LENGTH for ids in completion_ids]

    trainer_state = kwargs.get("trainer_state")
    _log(scored, truncated, getattr(trainer_state, "global_step", None))

    log_metric = kwargs.get("log_metric")
    if log_metric:
        usable = [scores for scores in scored if scores]
        if usable:
            for dimension in DIMENSIONS:
                log_metric(dimension, statistics.mean(s[dimension] for s in usable))
            # the judge-gaming detector: markers climbing while plausibility does
            # not means the policy is satisfying the rubric without sounding in
            # character. Watch its direction across training, not its value.
            log_metric(
                "gaming_gap",
                statistics.mean(s["markers"] for s in usable)
                - statistics.mean(s["human_plausibility"] for s in usable),
            )
        log_metric("judge_unscored", sum(scores is None for scores in scored))
        log_metric("truncated", sum(truncated))

    return [
        to_scalar(scores) if scores and not was_truncated else None
        for scores, was_truncated in zip(scored, truncated)
    ]
