# Checkpoint 2: RLAIF with GRPO

Takes the SFT checkpoint and optimizes it against the persona judge in [`judge/`](../judge)
using GRPO, with [slime](https://github.com/THUDM/slime) as the training framework.

| file | what it is |
| --- | --- |
| `install_slime.sh` | builds the slime environment (slime's `build_conda.sh`, repathed for a non-root user) |
| `slime_env.sh` | activates that environment; sourced by the other scripts |
| `merge_lora.py` | merges the SFT LoRA adapter into base weights, and checks the merge held |
| `prepare_prompts.py` | the SFT training prompts, without their responses, for rollouts |
| `convert_checkpoints.sh` | HF <-> Megatron `torch_dist`, the two conversions around training |
| `reward.py` | the judge as a GRPO reward |
| `test_reward.py` | offline checks, no network or API key |
| `run_grpo.sh` | the training launch |

## Running it

```bash
bash rlaif/install_slime.sh conda                 # once, ~15 min
bash rlaif/install_slime.sh deps                  # once, ~20 min of source builds

uv run python rlaif/prepare_prompts.py
uv run python rlaif/merge_lora.py
uv run python rlaif/merge_lora.py --verify        # gates the run, see below
bash rlaif/convert_checkpoints.sh to_megatron

bash rlaif/run_grpo.sh                            # needs OPENAI_API_KEY
bash rlaif/convert_checkpoints.sh to_hf outputs/rlaif/ckpt_gated/iter_0000040
```

Then evaluate the result through the checkpoint 1 harnesses, which take a full
checkpoint with `--model`:

```bash
uv run python evaluate_persona.py --model outputs/rlaif/rlaif_hf
uv run python evaluate_persona.py --score-only --judge --judge-model gpt-5.4-mini-2026-03-17
uv run python evaluation.py --model outputs/rlaif/rlaif_hf
```

Both harnesses merge into their cached results rather than overwriting, so the RLAIF
rows land beside the base and SFT rows they are being compared against. A full
checkpoint needs its own vLLM process and so cannot share a run with base + adapter.

## Why the reward is not the plain judge total

`judge/README.md` defines the reward as the 5-25 sum of the five dimensions. Scoring the
SFT checkpoint with the judge before training (EVAL_RESULTS.md section 4) showed why that
sum is the wrong thing to optimize:

- The SFT model already scores **20.48** against **20.62** for the human references. It is
  at the ceiling, and above the references on three of five dimensions.
- The headroom that remains is almost all in `markers` (1% of responses at ceiling) and
  `human_plausibility` (0% at ceiling). `markers` is the dimension the rubric itself flags
  as cheapest to satisfy superficially.
- `EXAMPLE D` in `judge_prompt.txt` is a marker-stuffed response the prompt scores
  `markers` 5 and `human_plausibility` 2. That totals **22**, above the SFT model's 20.59.

So an unweighted sum pays the policy to converge on the degenerate output the rubric
exists to detect. `REWARD_MODE=gated`, the default, scales the total by
`human_plausibility / 5`: the marker-stuffed response drops to 8.8 while an in-character
one holds 17.6. `human_plausibility` is also the only dimension where the SFT model is
*below* the human references, so it is the only direction left that is not overshoot.

`REWARD_MODE=sum` keeps the documented reward available for comparison.

## Decisions worth knowing about

**Full-parameter, not LoRA.** slime trains through Megatron and has no LoRA support, so
RLAIF starts from merged weights. `merge_lora.py --verify` confirms the merge is the same
model checkpoint 1 measured: style_cosine 0.877 against 0.862, judge total 20.55 against
20.48, every dimension within 0.05, and 89 of 200 responses byte-identical.

**One group per reward call.** `--group-rm` hands `reward.py` the whole group for a
prompt. That is what makes it possible to give an unscorable rollout its group's mean, so
its advantage is ~0. Flooring it instead would drag the group's mean down and distort its
siblings' advantages, not just its own.

**Truncated rollouts are not scored on their merits.** A rollout cut off at
`--rollout-max-response-len` loses `consistency` points for hitting the cap rather than
for anything the policy did. They are logged with a `truncated` flag and take the group
mean.

**The SFT policy is the KL reference.** `--ref-load` points at the SFT checkpoint, not the
base model, so the KL penalty measures drift from where checkpoint 1 finished. With the
persona already at the human ceiling, the risk being managed is drift, not underfitting.

**Training and evaluation share one judge.** Both import `judge.score_responses`. The
consequence is that the judge column alone cannot separate genuine improvement from
reward hacking, so the independent evidence is the stylometric table and the
`markers` minus `human_plausibility` gap. Watch the direction of that gap across
checkpoints: `outputs/rlaif/judge_scores_<mode>.jsonl` records every judged rollout.
