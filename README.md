# LoRA SFT for Qwen2.5-3B-Instruct with TRL

Fine-tunes the base model on the Rocky persona with a LoRA adapter, then measures both
what it gained (persona style) and what it cost (coding ability) against the untuned
baseline.

## Results at a glance

Baseline is vanilla `Qwen/Qwen2.5-3B-Instruct` with no adapter. Fine-tuned is
`outputs/sft/final_adapter`. Both are served through vLLM with greedy decoding, and both
get the same prompts and the same system prompt, so any difference comes from the
weights. Full methodology and caveats in [EVAL_RESULTS.md](EVAL_RESULTS.md).

### Persona transfer (200 held-out prompts)

Measured against the training corpus. The reference row is the held-out human responses
scored the same way, so it is what a perfect match looks like.

| | style cos | style dist | memorize |
| --- | --- | --- | --- |
| baseline (no adapter) | 0.585 | 2.698 | 0.000 |
| fine-tuned | **0.862** | **1.121** | 0.002 |
| reference (ceiling) | 0.981 | 1.062 | 0.001 |

style cos higher is better, style dist and memorize lower is better.

The adapter closes most of the gap to the target voice. Centroid distance drops from
2.70 to 1.12 against a 1.06 ceiling, so a typical response now sits about as close to the
persona as a real training example. Nothing is memorized.

The clearest single signal, since Rocky drops articles and copulas, is the rate of `the`
per 1k tokens: **61.5 baseline -> 4.9 fine-tuned**, against 2.5 for the references.

### Coding benchmarks (pass@1, full sets)

| benchmark | problems | baseline | fine-tuned | delta |
| --- | --- | --- | --- | --- |
| HumanEval | 164 | **0.726** | 0.555 | -0.171 |
| HumanEval+ | 164 | **0.695** | 0.506 | -0.189 |
| MBPP | 500 | **0.590** | 0.556 | -0.034 |

The persona was not free. The adapter also stopped emitting markdown code fences
entirely (164/164 baseline -> 0/164 fine-tuned) despite being asked for one, an
instruction-following regression the pass@1 numbers alone do not show.

### Where the artifacts live

| deliverable | path |
| --- | --- |
| training code | `train_sft.py`, `prepare_sft_data.py` |
| trained adapter | `outputs/sft/final_adapter/` (git-lfs) |
| raw generations, both models | `outputs/sft/persona_generations.json` |
| persona metrics | `outputs/sft/persona_eval_results.json` |
| benchmark scores | `outputs/sft/eval_results.json` |
| write-up | `EVAL_RESULTS.md` |
| persona judge rubric (checkpoint 2) | `judge/` |

## 1. Setup

```bash
uv sync
```

## 2. Data

One JSON object per line, either format:

```json
{"prompt": "Write a function that reverses a string.", "response": "```python\ndef reverse(s):\n    return s[::-1]\n```"}
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Kit, ..."}]}
```
Run the following command to convert the data into training format
```bash
uv run python prepare_sft_data.py
```

## 3. Train

```bash
uv run python train_sft.py
```

Full fine-tuning instead of LoRA: pass `peft_config=None` to `SFTTrainer` and lower `LEARNING_RATE`  This needs ~36GB ish

## 4. Output

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "outputs/sft/final_adapter")
```

## 5. Evaluate

Benchmarks checkpoints with vLLM for fast batched generation. Add a benchmark by writing a `run_xxx(llm, tokenizer, adapter_path, num_problems)` function in `evaluation.py` that returns a score, then adding it to `BENCHMARK_FUNCTIONS` and `BENCHMARKS`.

```bash
uv run python evaluation.py                                                   # all checkpoints, all benchmarks
uv run python evaluation.py --base                                            # base model only, no adapter
uv run python evaluation.py --checkpoints outputs/sft/checkpoint-100          # specific checkpoint(s)
uv run python evaluation.py --benchmarks mbpp --num-problems 300              # one benchmark, subsampled
uv run python evaluation.py --benchmarks bigcodebench                         # opt-in, first 200 of 1140
```

Results are written to `outputs/sft/eval_results.json`.

| benchmark | problems | in default run | dataset |
| --- | --- | --- | --- |
| `humaneval` | 164 | yes | `openai/openai_humaneval` |
| `humanevalplus` | 164 | yes | `evalplus/humanevalplus` |
| `mbpp` | 500 | yes | `google-research-datasets/mbpp` |
| `bigcodebench` | 200 of 1140 | no | `bigcode/bigcodebench` |

`humanevalplus` is the same 164 problems as `humaneval` with far more tests per
problem, and reuses its completions, so running both costs one generation pass. Read
them as a pair: the drop between them is the signal, since plain HumanEval's tests are
thin enough to pass solutions that are subtly wrong. `HumanEval/32` fails even its own
canonical solution, so 163/164 is the ceiling.

`bigcodebench` is library-heavy and opt-in, since executing its solutions needs
third-party packages:

```bash
uv sync --extra bigcodebench
uv run python evaluation.py --benchmarks bigcodebench --num-problems 1140      # full set
```

It defaults to 200 problems for iteration; use the full 1140 for any number you
report. The extras cover 1121 of the 1140 tasks; the rest need tensorflow, geopandas
or a pytesseract binary. Those and the ~60 network-dependent tasks fail uniformly
across checkpoints, so they lower the absolute score but not comparisons.

## 6. Persona style evaluation

Measures how closely outputs match the Rocky voice in the SFT dataset, so the base
model and the SFT model can be compared against the same target corpus. Prompts are
the 200 held-out examples in `data/raw/test_rocky.jsonl`; both models get the same
system prompt as training, so the persona has to come from the weights. The prepared
`data/persona_testing_data.jsonl` holds the same examples in prompt/completion form
and can be pointed at instead.

```bash
uv run python evaluate_persona.py --base                                      # base model only
uv run python evaluate_persona.py --base --checkpoints outputs/sft/final_adapter
uv run python evaluate_persona.py --score-only                                # rescore cached generations, no GPU
uv run python evaluate_persona.py --score-only --no-embeddings                # stylometry only, no extra deps
```

Four numbers per checkpoint:

| metric | what it measures | direction |
| --- | --- | --- |
| `style_cosine` | function-word + punctuation + rhythm profile vs. the training corpus profile | higher |
| `style_centroid_distance` | per-response distance to the training centroid, in std-units | lower |
| `embed_cosine_paired` | sentence-transformer cosine vs. the reference for the *same* prompt | higher |
| `embed_cosine_centroid` | sentence-transformer cosine vs. the training corpus centroid | higher |
| `memorization_jaccard` | highest 5-gram Jaccard against any single training example | lower |

The held-out references are scored too, as a `reference (ceiling)` row: that is what a
perfect persona match looks like, and it calibrates the rest of the table.

`memorization_jaccard` is the one column where the ceiling row is the floor: held-out
references score ~0.001, so anything approaching 1.0 means the checkpoint is
reproducing training examples verbatim, which every other metric here scores as a
perfect persona match. Responses above 0.5 are counted out under the table.

The stylometric numbers are the discriminative ones here. Rocky is defined by
grammatical surface features (`the` appears 2/1k tokens against 60/1k in normal
English, `a` 1/1k against 27/1k, plus heavy use of `good`/`listen`/`yes`), which is
exactly what function-word profiles capture.
The embedding numbers are dominated by topic, and topic is fixed by the prompt and so
is held constant across models, so expect them to move much less. The printed marker
rate table is the most legible evidence of transfer.

Generations are cached to `outputs/sft/persona_generations.json` and results to
`outputs/sft/persona_eval_results.json`; `--score-only` re-reads the cache so metrics
can be iterated on without a GPU.

## 7. AI judge for persona adherence (checkpoint 2, RLAIF)

`judge/` holds the rubric that scores persona adherence with an LLM judge: five
dimensions, 1 to 5 each, summed into a 5-25 scalar for the GRPO reward. It is the
qualitative counterpart to the stylometric metrics in section 6, and the `markers` vs.
`human_plausibility` gap is what answers the assignment's "did the AI judge get gamed?"
question.

| file | what it is |
| --- | --- |
| `judge/judge_prompt.txt` | prompt template, `{STYLE_GUIDE}` and `{RESPONSE_PROSE}` placeholders |
| `judge/rocky_style.md` | the style guide substituted into `{STYLE_GUIDE}` |
| `judge/scorer.py` | the shared implementation the reward and the eval both import |
| `judge/README.md` | how to run it, and the open item to settle before the GRPO run |

Scoring is provider-agnostic: `score_response` takes a `complete` callable, `str -> str`.
The GRPO loop imports `reward()`, the eval imports `score_responses()`, so both stages
use one instrument and the RLAIF-vs-SFT comparison stays honest.

```bash
uv run python judge/test_scorer.py                                            # offline, no API key
uv run python evaluate_persona.py --score-only --judge --judge-model claude-sonnet-5
```

`--judge` runs off the cached generations, so it needs no GPU. It adds a `judge 5-25`
column to the table in section 6, a per-dimension breakdown, and the `markers` minus
`human_plausibility` gap. Run it on the SFT generations before GRPO starts to get the
checkpoint 1 baseline. See [judge/README.md](judge/README.md) for the full notes.
