# LoRA SFT for Qwen2.5-3B-Instruct with TRL

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

The held-out references are scored too, as a `reference (ceiling)` row: that is what a
perfect persona match looks like, and it calibrates the rest of the table.

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
