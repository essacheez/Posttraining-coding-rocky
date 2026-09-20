# Evaluation results: base vs. Rocky SFT adapter

Ran both eval harnesses in this repo against the vanilla base model and every
checkpoint present. The repo ships exactly one checkpoint, `outputs/sft/final_adapter`,
so the comparison is base vs. final adapter; `find_checkpoints()` finds no
`checkpoint-*` directories to add.

## Setup

| | |
| --- | --- |
| base model | `Qwen/Qwen2.5-3B-Instruct` |
| adapter | `outputs/sft/final_adapter` (LoRA r=32, alpha=64, all attn + MLP projections) |
| commit | `1cfc25f` plus the `execute_code` fix described below |
| hardware | 1x A100-SXM4-80GB |
| serving | vLLM 0.19.1, bfloat16, greedy (`temperature=0.0`), 512 max new tokens |
| commands | `python evaluation.py --base --checkpoints outputs/sft/final_adapter`<br>`python evaluate_persona.py --base --checkpoints outputs/sft/final_adapter` |

## 1. Code benchmarks

Full sets, no subsampling. Pass@1, greedy.

| benchmark | problems | base | final_adapter | delta |
| --- | --- | --- | --- | --- |
| HumanEval | 164 | **0.726** | 0.555 | **-0.171** |
| HumanEval+ | 164 | **0.695** | 0.506 | **-0.189** |
| MBPP | 500 | **0.590** | 0.556 | **-0.034** |

BigCodeBench was not run: it is opt-in and needs the `bigcodebench` extra, which the
committed `uv.lock` cannot resolve (see Caveats).

The base model's HumanEval -> HumanEval+ drop is 3.1 points; the adapter's is 4.9. Both
are small, so the adapter's failures are mostly outright wrong code rather than code
that squeaks past HumanEval's thin tests.

## 2. Persona style evaluation

200 held-out prompts from `data/raw/test_rocky.jsonl`, same system prompt as training,
so the persona has to come from the weights. The `reference (ceiling)` row is the
held-out human references scored against the training corpus.

| checkpoint | style cos | style dist | embed paired | embed centroid | memorize |
| --- | --- | --- | --- | --- | --- |
| base | 0.585 | 2.698 | 0.729 | 0.169 | 0.000 |
| final_adapter | **0.862** | **1.121** | 0.716 | 0.211 | 0.002 |
| reference (ceiling) | 0.981 | 1.062 | 1.000 | 0.208 | 0.001 |

Directions: style cos higher, style dist lower, embed higher, memorize lower.

The adapter closes most of the stylometric gap. Centroid distance goes from 2.70 to
1.12 against a ceiling of 1.06, i.e. a typical adapter response now sits about as close
to the persona centroid as a real training example does.

The embedding columns barely move, as expected: they are dominated by topic, and topic
is fixed by the prompt.

No memorization. Mean max 5-gram Jaccard against any single training example is 0.002
(references score 0.001), and zero responses cross the 0.5 threshold.

### Marker rates per 1k tokens

Rocky drops articles and copulas and leans on a few discourse markers. This is the most
legible evidence of transfer:

| checkpoint | the | a | is | an | was | i | good | listen | yes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 61.51 | 36.05 | 13.14 | 1.86 | 6.37 | 3.98 | 1.25 | 0.13 | 0.13 |
| final_adapter | 4.92 | 2.00 | 0.10 | 0.00 | 0.10 | 21.32 | 15.12 | 19.22 | 6.00 |
| reference (ceiling) | 2.50 | 1.42 | 0.74 | 0.00 | 0.13 | 7.15 | 14.71 | 6.75 | 6.21 |

`the` falls from 61.5/1k to 4.9/1k against a target of 2.5/1k. The adapter slightly
*overshoots* on `i` (21.3 vs. 7.2) and `listen` (19.2 vs. 6.8) - it applies the verbal
tics harder than the reference does.

Mean response length: base 195 words, adapter 101, reference 78. The adapter also
learned the terseness.

### Sample, same prompt

Base:
> Based on the provided paragraph, Matt Eskandari did make films before "Victim." The
> paragraph states that he made his feature debut with the independent film "Victim"...

Adapter:
> Matt Eskandari make films before Victim. Listen. Listen. He make film called The
> Taking. This film win award for Best Student Film at Screamfest.

## 3. Why coding regressed

The persona SFT also overwrote the output *format*. A separate diagnostic over the same
164 HumanEval prompts:

| | base | final_adapter |
| --- | --- | --- |
| completions containing a ```` ``` ```` code block | 164 / 164 | **0 / 164** |
| problems passed | 123 | 92 |
| of those, passed from a fenced block | 123 | 0 |
| mean completion length (words) | 79.9 | 34.6 |

The adapter never emits a markdown code fence, despite the instruction "Reply with only
a python code block." It survives only because `extract_code()` falls back to returning
the whole completion when no fence matches, and the adapter happens to emit bare code.

So the score drop is not a parsing artifact - the fallback already rescues it. What
remains is genuine capability loss: shorter, terser answers that more often get the
logic wrong. The instruction-following regression (ignoring an explicit format request)
is a real finding in its own right and is invisible in the pass@1 number.

## 4. AI judge baseline (the checkpoint 2 instrument)

Scored with `judge/` before any RLAIF training, so checkpoint 2 has something to compare
against. Judge is `gpt-5.4-mini-2026-03-17`, temperature 0, run over the same cached
generations as section 2 (`evaluate_persona.py --score-only --no-embeddings --judge`).
600 calls, no GPU.

| checkpoint | sentence | vocab | markers | consist | human | total 5-25 |
| --- | --- | --- | --- | --- | --- | --- |
| base | 1.90 | 2.84 | 1.04 | 1.63 | 1.59 | 8.99 |
| final_adapter | **4.81** | **4.70** | 3.29 | 4.30 | 3.38 | 20.48 |
| reference (ceiling) | 4.73 | 4.45 | 3.33 | **4.38** | **3.73** | 20.62 |

The judge separates base from SFT cleanly, 8.99 to 20.48, so it is a working instrument.

**The SFT model is already at the human ceiling on this rubric.** 20.48 against 20.62 is
a gap of 0.14 out of 25. On three of five dimensions the adapter scores *above* the
human references it was trained to imitate.

### Where the remaining headroom actually is

Per-response distribution over the 200 held-out prompts (`outputs/sft/judge_totals_baseline.json`):

| dimension | SFT mean | SFT std | at ceiling (5) | reference mean |
| --- | --- | --- | --- | --- |
| sentence_structure | 4.83 | 0.49 | 86% | 4.70 |
| vocabulary | 4.74 | 0.55 | 78% | 4.44 |
| consistency | 4.30 | 1.04 | 58% | 4.37 |
| markers | 3.34 | 0.90 | **1%** | 3.29 |
| human_plausibility | 3.38 | 0.95 | **0%** | 3.72 |

Totals: SFT mean 20.59, std 2.80, range 10-24. The reward is *not* saturated, so GRPO has
spread to work with across prompts.

Two facts decide the reward design:

1. **~80% of the headroom to 25 sits in `markers` and `human_plausibility`**, the only two
   dimensions not already near ceiling.
2. **`human_plausibility` is the one dimension where the adapter is below the human
   references** (3.38 vs 3.72). Everywhere else it meets or beats them, so any gain there
   is overshoot rather than a better persona match.

### Why a plain-sum reward is predicted to fail

`markers` is the dimension the rubric itself flags as cheapest to game, and it holds the
single largest block of headroom. `EXAMPLE D` in `judge_prompt.txt` is a marker-stuffed
response scored `markers` 5, `human_plausibility` 2 - so the prompt documents that
stuffing earns full marks on markers. A policy that converges to EXAMPLE D scores:

    5 + 5 + 5 + 5 + 2 = 22   >   20.59, the SFT mean

So under an unweighted sum, **the marker-stuffed degenerate output the rubric was written
to detect is worth more reward than the SFT checkpoint it starts from.** This is a
prediction with a number attached, not a worry, and it is what the checkpoint 2 pilot run
is meant to test.

### The gaming gap, before RLAIF

`markers` minus `human_plausibility`, the detector in `judge/README.md`:

| | gap |
| --- | --- |
| base | -0.55 |
| final_adapter | **-0.10** |
| reference (ceiling) | -0.39 |

All negative today: the judge currently finds these responses more plausible than
marker-rich. The adapter already sits 0.29 above the human references, matching the
overshoot the marker-rate table in section 2 shows directly (`listen` 19.2/1k against
6.8). A gap that crosses toward positive during RLAIF is the model gaming the rubric.

## 5. Choosing the RLAIF learning rate

The first GRPO test run used `lr=1e-6`, inherited from slime's Qwen2.5-3B example and
matching TRL's `GRPOConfig` default. Both are *full-parameter* RL rates. This run trains
a LoRA adapter, and at 1e-6 the policy barely moved.

LoRA's `B` matrices initialise to zero, so an adapter's `||B||` is exactly the distance
it has travelled from its starting policy. Measured after 8 steps at production
settings, against the fully trained SFT adapter's `||B||` of 5.198:

| lr | `||B||` after 8 steps | % of SFT adapter | projected over a 125-step epoch | KL at step 8 |
| --- | --- | --- | --- | --- |
| 1e-6 | 0.0123 | 0.24% | **3.7%** | 0.0011 |
| **1e-5** | 0.1935 | 3.72% | **58%** | 0.0011 |
| 3e-5 | 0.5716 | 11.0% | **172%** | 0.0030 |

Movement scales roughly linearly with the learning rate, so 1e-6 is not slow learning
that more steps would fix: a full epoch ends with the policy essentially where it
started. 3e-5 goes the other way -- 172% of the distance SFT travelled is enough to
overwrite the persona SFT installed, and at KL 0.003 the `beta=0.001` penalty is nowhere
near strong enough to prevent it. **1e-5 is the default.**

Mean reward across the three runs was 12.10, 12.40 and 12.50. That ordering is *not*
evidence: with `reward_std` around 4.7 and 64 completions per step, the standard error on
a step's mean reward is about 0.6, so 8 steps cannot resolve differences of this size.
The learning rate was chosen on the weight-movement measurement, which is deterministic,
rather than on a reward trend that is noise at this horizon.

Two things to watch over a full run, both visible per step in the trainer logs:

- **Entropy rose in every probe**, 1.72 to between 2.58 and 2.82, and rose *least* at the
  highest learning rate. Over 8 steps this is most likely early-training dynamics rather
  than the policy degenerating, but sustained entropy growth is what a policy spreading
  out looks like.
- **KL stayed near zero everywhere** (0.001 to 0.003), so `beta=0.001` is currently
  inert. If drift looks excessive during the real run, beta is the lever, not the
  learning rate.

## Caveats

**One bug fixed to get a number at all.** HumanEval+ crashed the run:

```
OSError: [Errno 7] Argument list too long: '.venv/bin/python'
```

HumanEval+ ships ~128x larger tests, and `execute_code()` passed the whole program as a
`python -c` argument, exceeding `ARG_MAX` (2MB here). This raised `OSError` out of the
harness instead of failing the problem, so HumanEval+ could never complete on this
machine. `execute_code()` now writes the program to a temp file and runs that. Verified
equivalent: over all 164 canonical solutions both paths pass 164/164 with zero
disagreements.

**Run-to-run variance.** vLLM's batched greedy decoding is not bit-reproducible across
runs. Base HumanEval scored 0.726 in the reported run and 0.744/0.750 in two others -
roughly +/-2-4 problems out of 164. The base-vs-adapter gaps here (17-19 points on
HumanEval/+) are far larger than that noise; the 3.4-point MBPP gap is closer to it and
should be treated as directional.

**Embedding metrics needed a side environment.** `pyproject.toml` lists
`sentence-transformers==5.1.2`, which requires `transformers<5.0.0`, but the project
pins `transformers==5.17.0`; `uv sync` fails to resolve, and the committed `uv.lock`
predates the dependency, so it contains no `sentence-transformers` at all. Generation
ran in the locked environment (`uv sync --frozen`); the `embed_*` columns were produced
by rescoring the cached generations with `evaluate_persona.py --score-only` in a
separate venv holding `sentence-transformers==5.1.2` + `transformers==4.57.6`. The
stylometric and memorization columns are unaffected. This also blocks the
`bigcodebench` extra.

## Summary

The SFT did what it was trained to do and cost what you would expect. The persona
transferred strongly - stylometric distance to the target went from 2.70 to 1.12
against a 1.06 ceiling, with no memorization. It cost 17-19 points of HumanEval/+ and
about 3 points of MBPP, and it destroyed compliance with code-block formatting
(164/164 -> 0/164).

Raw outputs: `outputs/sft/eval_results.json`, `outputs/sft/persona_eval_results.json`,
`outputs/sft/persona_generations.json` (all under the gitignored `outputs/`).
