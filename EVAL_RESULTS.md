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
