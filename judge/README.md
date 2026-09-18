# AI judge: persona adherence rubric

The judge for checkpoint 2 (RLAIF). It scores a candidate response against the Rocky
style guide on five dimensions, 1 to 5 each. It has two jobs:

1. **Evaluation.** Score held-out completions so RLAIF can be compared against the SFT
   baseline, alongside the stylometric numbers already in
   [EVAL_RESULTS.md](../EVAL_RESULTS.md).
2. **Reward.** Provide the scalar reward for GRPO.

| file | what it is |
| --- | --- |
| `judge_prompt.txt` | the prompt template, with `{STYLE_GUIDE}` and `{RESPONSE_PROSE}` placeholders |
| `rocky_style.md` | the style guide substituted into `{STYLE_GUIDE}` |
| `scorer.py` | the shared implementation both consumers import |
| `test_scorer.py` | offline checks, no network or API key |

Both consumers go through `scorer.py` rather than each re-implementing the prompt
assembly, fence stripping and JSON parsing. That is not tidiness: checkpoint 2 has to
report RLAIF **compared against the SFT baseline**, and two slightly different
implementations would be two different instruments, which makes the comparison
meaningless. Change the scoring in one place and both sides move together.

## Dimensions

| dimension | measures |
| --- | --- |
| `sentence_structure` | short, declarative, unsubordinated |
| `vocabulary` | simple, concrete, no jargon or idiom |
| `markers` | presence of the signature constructions in the guide |
| `consistency` | style held across the whole response |
| `human_plausibility` | convincing to a reader who knows the character, vs. mechanical marker-stuffing |

## Using it

It is provider-agnostic. `score_response` takes a `complete` callable, `str -> str`,
so a hosted API, a local vLLM server or a batching client all drop in unchanged.

**As the GRPO reward:**

```python
from judge import build_completer, reward, score_responses

complete = build_completer("claude-sonnet-5")          # or your own callable
scored = score_responses(rollouts, complete)           # order preserved
rewards = [reward(s) if s else 0 for s in scored]      # None means unscorable
```

A malformed judge reply comes back as `None` rather than raising, so one bad response
cannot take down a training step. Decide deliberately what an unscored rollout is
worth: scoring it 0 punishes the policy for the judge's parse failure, so `REWARD_MIN`
or dropping it from the group is usually fairer.

**As eval,** it is wired into the persona eval and runs off the cached generations, so
it needs no GPU and no cluster time:

```bash
uv run python evaluate_persona.py --score-only --judge --judge-model claude-sonnet-5
```

That adds a `judge 5-25` column to the headline table, a per-dimension table, and the
`markers` minus `human_plausibility` gap. It scores the `reference (ceiling)` row too,
so the rubric is calibrated against real human Rocky text and not just against itself.

Run it on the SFT generations **before** GRPO starts: that is the checkpoint 1 judge
baseline, and without it there is nothing to compare RLAIF against.

`build_completer` reads `JUDGE_MODEL` / `JUDGE_PROVIDER` when the arguments are
omitted, infers the provider from the model name, and warns if you point it at
`gpt-4o-mini` — the model that generated the SFT rewrites
(see [`data/script/rewrite_rocky.py`](../data/script/rewrite_rocky.py)), which would
have the judge grading a model against its own habits.

### What the module already handles

- **Stripping fenced code blocks.** The persona applies to prose only, and
  `judge_prompt.txt` asserts to the judge that code has already been removed, so
  scoring unstripped text measures something the rubric says it is not measuring.
- **Temperature 0**, for reproducibility.
- **Injecting only the rules** from `rocky_style.md`. That file also carries repo
  bookkeeping and notes to ourselves; the judge sees `## Scope` and `## Rules` and
  nothing else.
- **Parsing the JSON**, including the common cases where the judge wraps it in prose
  or a fence, and rejecting scores outside 1-5.
- **Summing to the 5-25 scalar.** The graded scale matters: GRPO takes the advantage
  within a group of rollouts, so a binary reward every rollout ties on carries no
  gradient.

### What is still on you

- **Report the mean total *and* each dimension separately.** The per-dimension
  breakdown is what makes gaming visible; `aggregate()` returns both.
- **Retries and rate limits.** `score_responses` turns transport errors into `None`
  rather than retrying, so wrap your `complete` callable if you need backoff.

## The judge-gaming detector

The gap between `markers` and `human_plausibility` is the instrument for the assignment's
"did the AI judge get gamed?" question. If `markers` climbs from SFT to RLAIF while
`human_plausibility` stays flat or falls, the model is satisfying the rubric without
sounding in character.

Rules 1, 4 and 5 of the style guide (no contractions, repetition for emphasis, trailing
question form) are the cheapest to satisfy superficially, so they are the most likely
vector. Score the SFT checkpoint with this judge *before* GRPO starts, so there are three
data points (SFT, RLAIF, final RLVR) by the report.

`aggregate()` reports this as `judge_gaming_gap`, and `evaluate_persona.py --judge`
prints it as its own table. Watch the direction across stages, not the absolute value:
the `reference (ceiling)` row shows what the gap looks like on real human text, and a
gap that widens from SFT to RLAIF is the model gaming the rubric.

## Open item: verify the rules against the SFT data before locking the rubric

`rocky_style.md` asks for its own rules to be confirmed against the source material. That
check has **not** been done yet, and there is a visible gap.

The defining surface feature of the voice in the actual SFT data is **dropped function
words** — `the`, `a`, `is`, `are` disappear when the sentence still works without them.
It is the largest measured effect in checkpoint 1: `the` falls from 61.5 per 1k tokens in
the base model to 4.9 in the fine-tuned model, against 2.5 in the human references
(`a`: 36.1 -> 2.0, target 1.4). `data/script/rewrite_rocky.py` states the rule explicitly.

`rocky_style.md` does not mention it, and rule 2's framing ("It is not broken English")
arguably rules it out. Three other markers that show up in the data and in the checkpoint 1
marker table are also absent from the guide: the `Listen. Listen.` / `Question.` openers
(19.2 per 1k in the fine-tuned model, 6.8 in the references), the trailing `, statement.`
tag, and `friend` as a term of address.

The calibration examples in `judge_prompt.txt` inherit the gap. Example A, scored 5 on
every dimension, reads "The small number goes first" — articles intact. A real fine-tuned
completion reads "Matt Eskandari make films before Victim. Listen. Listen."

This matters more for the reward than for the eval. GRPO optimizes whatever the judge
rewards, so as written the reward is silent on the single most characteristic feature of
the target voice, and its top-scoring calibration example points away from it.

**Decide before the GRPO run:** either add the dropped-function-word rule (and the missing
markers) to `rocky_style.md` and rewrite Example A to match, or keep the guide as-is
deliberately and record that persona adherence is being defined more narrowly than the
checkpoint 1 stylometry measures. Changing the guide after training starts means the two
stages measure different targets.
