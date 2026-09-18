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

## Dimensions

| dimension | measures |
| --- | --- |
| `sentence_structure` | short, declarative, unsubordinated |
| `vocabulary` | simple, concrete, no jargon or idiom |
| `markers` | presence of the signature constructions in the guide |
| `consistency` | style held across the whole response |
| `human_plausibility` | convincing to a reader who knows the character, vs. mechanical marker-stuffing |

## Usage

- **Strip fenced code blocks** from the model output before substituting into
  `{RESPONSE_PROSE}`. The persona applies to prose only; code blocks are left in
  standard formatting, which is what keeps persona and capability separable for the
  final tradeoff analysis.
- **Run the judge at temperature 0** for reproducibility.
- **Use a different model family** than whatever generates the SFT rewrites, so the
  judge is not grading a model against its own habits. The SFT rewrites came from
  `gpt-4o-mini` (see [`data/script/rewrite_rocky.py`](../data/script/rewrite_rocky.py)).
- **Parse the JSON and report the mean total *and* each dimension separately.** The
  per-dimension breakdown is what makes gaming visible.
- **For the GRPO reward, sum the five dimensions into a 5-25 scalar.** The graded scale
  matters: GRPO computes advantage within a group of rollouts, so a binary reward where
  every rollout scores the same gives zero advantage and no learning signal.

## The judge-gaming detector

The gap between `markers` and `human_plausibility` is the instrument for the assignment's
"did the AI judge get gamed?" question. If `markers` climbs from SFT to RLAIF while
`human_plausibility` stays flat or falls, the model is satisfying the rubric without
sounding in character.

Rules 1, 4 and 5 of the style guide (no contractions, repetition for emphasis, trailing
question form) are the cheapest to satisfy superficially, so they are the most likely
vector. Score the SFT checkpoint with this judge *before* GRPO starts, so there are three
data points (SFT, RLAIF, final RLVR) by the report.

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
