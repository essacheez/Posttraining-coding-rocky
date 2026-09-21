# Post-training Qwen2.5-3B-Instruct on the Rocky persona

Post-training pipeline for the Rocky persona (the Eridian from *Project Hail Mary*)
on `Qwen/Qwen2.5-3B-Instruct`, plus the harness that measures what each stage gained
in persona style against what it cost in coding ability.

## Repo layout

| folder | what it holds |
| --- | --- |
| [`SFT/`](SFT/) | checkpoint 1: LoRA supervised fine-tuning. Training code, data prep, raw data, and the trained adapter |
| [`RLAIF/`](RLAIF/) | checkpoint 2: GRPO with an LLM judge reward. Training code, the GRPO prompt dataset and its build script |
| `RLVR/` | checkpoint 3: RL with verifiable rewards (planned, not yet implemented) |
| [`EVALS/`](EVALS/) | evaluation and plotting code shared across checkpoints |

Each of `SFT/` and `RLAIF/` is self-contained: a `training/` folder with the trainer,
a `data/` folder with data prep and the dataset itself, and a `model/` folder with the
trained checkpoints (gitignored except for the final adapter, which is force-tracked
via git-lfs).

## Setup

```bash
uv sync
```

## Results at a glance (checkpoint 1: SFT)

Baseline is vanilla `Qwen/Qwen2.5-3B-Instruct` with no adapter. Fine-tuned is
`SFT/model/final_adapter`. Both are served through vLLM with greedy decoding, and both
get the same prompts and the same system prompt, so any difference comes from the
weights. Full methodology and caveats in [EVALS/EVAL_RESULTS.md](EVALS/EVAL_RESULTS.md).

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
| training code | `SFT/training/train_sft.py`, `SFT/data/prepare_sft_data.py` |
| trained adapter | `SFT/model/final_adapter/` (git-lfs) |
| raw generations, both models | `EVALS/results/persona_generations.json` |
| persona metrics | `EVALS/results/persona_eval_results.json` |
| benchmark scores | `EVALS/results/eval_results.json` |
| write-up | `EVALS/EVAL_RESULTS.md` |

## SFT (checkpoint 1)

### 1. Data

One JSON object per line, either format:

```json
{"prompt": "Write a function that reverses a string.", "response": "```python\ndef reverse(s):\n    return s[::-1]\n```"}
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Kit, ..."}]}
```

Run the following command to convert the data into training format:

```bash
uv run python SFT/data/prepare_sft_data.py
```

### 2. Train

```bash
uv run python SFT/training/train_sft.py
```

Full fine-tuning instead of LoRA: pass `peft_config=None` to `SFTTrainer` and lower `LEARNING_RATE`. This needs ~36GB ish.

### 3. Output

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "SFT/model/final_adapter")
```

## RLAIF (checkpoint 2)

GRPO on top of the SFT model, with an LLM judge as the reward.

### 1. Data

500 new prompts from the same three sources as the SFT data (No Robots, Dolly-15k,
OASST2), none of them in the SFT set.

```bash
uv run python RLAIF/data/build_grpo_dataset.py
```

Writes `RLAIF/data/grpo_dataset.jsonl`.

### 2. Train

Merge the SFT adapter into the base model first, then run GRPO on top of it:

```bash
uv run python RLAIF/training/merge_adapter.py
uv run python RLAIF/training/train_rlaif.py
```

The reward is an LLM judge that scores each rollout on task correctness (0 to 5) and
Rocky style (1 to 5). The judge prompt is in `RLAIF/training/judge.txt`. Put
`OPENAI_API_KEY` in a `.env` at the repo root.

### 3. Output

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("RLAIF/model/sft_merged")
model = PeftModel.from_pretrained(base, "RLAIF/model/checkpoint-45")
```

### 4. Results

Base is the merged SFT model with no adapter. Same prompts and decoding as the SFT eval.

| checkpoint | style dist | HumanEval | HumanEval+ | MBPP |
| --- | --- | --- | --- | --- |
| base (sft_merged) | 1.183 | 0.579 | 0.543 | 0.554 |
| checkpoint-15 | 1.083 | 0.591 | 0.537 | 0.552 |
| checkpoint-45 | 1.073 | 0.585 | 0.530 | 0.562 |
| checkpoint-75 | **1.003** | 0.067 | 0.061 | 0.008 |
| checkpoint-105 | 1.034 | 0.159 | 0.152 | 0.044 |
| checkpoint-112 | 1.056 | 0.146 | 0.134 | 0.018 |

style dist lower is better. Coding is pass@1 on the full sets.

Persona gets better through checkpoint-75, where a response sits as close to the persona
as a real training example. Coding falls off a cliff between 45 and 75. The judge only
scores the prose around code, so the model learns to narrate every answer step by step
and split the code into several fenced blocks. The eval reads only the first block, so
the function under test is missing and almost every problem fails. This will be fixed in the next submission. 

## RLVR (checkpoint 3)

Planned: RL with verifiable (non-AI-judge) rewards. Not yet implemented.

## Evaluate

Benchmarks checkpoints with vLLM for fast batched generation. Add a benchmark by writing a `run_xxx(llm, tokenizer, adapter_path, num_problems)` function in `EVALS/evaluation.py` that returns a score, then adding it to `BENCHMARK_FUNCTIONS` and `BENCHMARKS`.

```bash
uv run python EVALS/evaluation.py                                                   # all checkpoints, all benchmarks
uv run python EVALS/evaluation.py --base                                            # base model only, no adapter
uv run python EVALS/evaluation.py --checkpoints SFT/model/checkpoint-100            # specific checkpoint(s)
uv run python EVALS/evaluation.py --benchmarks mbpp --num-problems 300              # one benchmark, subsampled
uv run python EVALS/evaluation.py --benchmarks bigcodebench                         # opt-in, first 200 of 1140
uv run python EVALS/evaluation.py --benchmarks livecodebench                        # opt-in, all 175 problems
```

Results are written to `EVALS/results/eval_results.json`.

| benchmark | problems | in default run | dataset |
| --- | --- | --- | --- |
| `humaneval` | 164 | yes | `openai/openai_humaneval` |
| `humanevalplus` | 164 | yes | `evalplus/humanevalplus` |
| `mbpp` | 500 | yes | `google-research-datasets/mbpp` |
| `bigcodebench` | 200 of 1140 | no | `bigcode/bigcodebench` |
| `livecodebench` | 175 | no | `livecodebench/code_generation_lite` (`test6.jsonl`) |

`humanevalplus` is the same 164 problems as `humaneval` with far more tests per
problem, and reuses its completions, so running both costs one generation pass. Read
them as a pair: the drop between them is the signal, since plain HumanEval's tests are
thin enough to pass solutions that are subtly wrong. `HumanEval/32` fails even its own
canonical solution, so 163/164 is the ceiling.

`bigcodebench` is library-heavy and opt-in, since executing its solutions needs
third-party packages:

```bash
uv sync --extra bigcodebench
uv run python EVALS/evaluation.py --benchmarks bigcodebench --num-problems 1140      # full set
```

It defaults to 200 problems for iteration; use the full 1140 for any number you
report. The extras cover 1121 of the 1140 tasks; the rest need tensorflow, geopandas
or a pytesseract binary. Those and the ~60 network-dependent tasks fail uniformly
across checkpoints, so they lower the absolute score but not comparisons.

`livecodebench` is competitive-programming style (stdin/stdout, plus LeetCode-style
`Solution` classes) rather than function completion, and pass@1 requires every private
test case for a problem to pass (public test cases are not used, they are the easy,
often hand-written subset). It only loads `test6.jsonl` (release_v6, contest dates
2025-01 through 2025-04), the most recent shard and the one least likely to be in a
base model's pretraining data. `--num-problems` still subsamples if you want to iterate
faster, most-recently-dated first.

## Persona style evaluation

Measures how closely outputs match the Rocky voice in the SFT dataset, so the base
model and the SFT model can be compared against the same target corpus. Prompts are
the 200 held-out examples in `SFT/data/raw/test_rocky.jsonl`; both models get the same
system prompt as training, so the persona has to come from the weights. The prepared
`SFT/data/persona_testing_data.jsonl` holds the same examples in prompt/completion form
and can be pointed at instead.

```bash
uv run python EVALS/evaluate_persona.py --base                                      # base model only
uv run python EVALS/evaluate_persona.py --base --checkpoints SFT/model/final_adapter
uv run python EVALS/evaluate_persona.py --score-only                                # rescore cached generations, no GPU
uv run python EVALS/evaluate_persona.py --score-only --no-embeddings                # stylometry only, no extra deps
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

Generations are cached to `EVALS/results/persona_generations.json` and results to
`EVALS/results/persona_eval_results.json`; `--score-only` re-reads the cache so metrics
can be iterated on without a GPU.
