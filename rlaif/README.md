# Checkpoint 2: RLAIF with GRPO

Takes the SFT checkpoint and optimizes it against the persona judge in
[`judge/`](../judge) using GRPO, with TRL — the same framework as checkpoint 1.

| file | what it is |
| --- | --- |
| `merge_lora.py` | merges a LoRA adapter into its base weights, and checks the merge held |
| `prepare_prompts.py` | the SFT training prompts, without their responses, for rollouts |
| `reward.py` | the judge as a GRPO reward |
| `test_reward.py` | offline checks, no network, API key or GPU |
| `train_grpo.py` | the training run |

## Running it

```bash
uv run python rlaif/prepare_prompts.py
uv run python rlaif/merge_lora.py                 # SFT adapter -> outputs/sft/merged_hf
uv run python rlaif/merge_lora.py --verify        # gates the run, see below

OPENAI_API_KEY=... uv run python rlaif/train_grpo.py
```

Then flatten the trained adapter and evaluate it through the checkpoint 1 harnesses:

```bash
uv run python rlaif/merge_lora.py \
    --base outputs/sft/merged_hf \
    --adapter outputs/rlaif/grpo_gated/final_adapter \
    --output outputs/rlaif/rlaif_hf

uv run python evaluate_persona.py --model outputs/rlaif/rlaif_hf
uv run python evaluate_persona.py --score-only --judge --judge-model gpt-5.4-mini-2026-03-17
uv run python evaluation.py --model outputs/rlaif/rlaif_hf
```

Both harnesses merge into their cached results rather than overwriting, so the RLAIF
rows land beside the base and SFT rows they are compared against. A full checkpoint
needs its own vLLM process and so cannot share a run with base + adapter.

## Why the reward is not the plain judge total

`judge/README.md` defines the reward as the 5-25 sum of the five dimensions. Scoring the
SFT checkpoint with the judge before training ([EVAL_RESULTS.md](../EVAL_RESULTS.md)
section 4) showed why that sum is the wrong thing to optimize:

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

**Train a fresh adapter over the merged SFT weights.** `train_grpo.py` loads
`outputs/sft/merged_hf` and attaches a new LoRA adapter rather than continuing the SFT
adapter. That is what makes the KL reference right: TRL computes reference logprobs with
the adapter disabled, which is the SFT policy, so `beta` measures drift from where
checkpoint 1 finished rather than from the base model. Continuing the SFT adapter would
have silently anchored the KL to base Qwen instead.

`merge_lora.py --verify` confirms the merge is the model checkpoint 1 measured:
style_cosine 0.877 against 0.862, judge total 20.55 against 20.48, every dimension within
0.05, and 89 of 200 responses byte-identical. It also probes whether GRPO has a gradient
at all — within-group reward std was 2.36 with no flat groups, so there is signal to climb.

**Unscorable and truncated rollouts return `None`.** TRL excludes a `None` from the
group's baseline and forces that completion's advantage to 0. Returning a floor instead
would pull the group mean down and distort its siblings' advantages, not just its own. A
completion that hit `max_completion_length` is treated this way because the judge marks it
down on `consistency` for hitting the cap rather than for anything the policy did.

**Training and evaluation share one judge.** Both import `judge.score_responses`. The
consequence is that the judge column alone cannot separate genuine improvement from
reward hacking, so the independent evidence is the stylometric table and the `markers`
minus `human_plausibility` gap. The reward logs both through TRL's `log_metric`, so
`gaming_gap` is plotted per step alongside `kl` and `entropy`; every judged rollout is
also written to `outputs/rlaif/judge_scores_<mode>.jsonl`.

## An earlier attempt at this used slime

The first version of this directory ran GRPO through
[slime](https://github.com/THUDM/slime) (SGLang rollouts, Megatron training). It worked
as far as a launchable configuration, and the environment builds, but it cost four extra
moving parts that TRL does not need: a from-source install of SGLang, Megatron and apex;
full-parameter training, since slime has no LoRA support; HF to Megatron `torch_dist`
conversion in both directions; and a separate conda environment from the rest of the
repo. TRL reuses checkpoint 1's environment, framework and LoRA setup, and its GRPO
implementation already handles unscorable completions the way the reward needs.

The slime version is in this branch's history if it is ever wanted back.
