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

The last turn must be the assistant. Rows without a system message get `SYSTEM_PROMPT`. Set `INPUT_FILES` at the top of `prepare_sft_data.py`, then:

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
```

Results are written to `outputs/sft/eval_results.json`.
