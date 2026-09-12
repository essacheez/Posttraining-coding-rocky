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

Output rows are `{"prompt": [all turns but the last], "completion": [last assistant turn]}`; TRL computes the loss on the completion only.

## 3. Train

```bash
uv run python train_sft.py                              # one GPU
uv run torchrun --nproc_per_node 2 train_sft.py         # two GPUs, data parallel
```

Full fine-tuning instead of LoRA: pass `peft_config=None` to `SFTTrainer` and lower `LEARNING_RATE` to about `1e-5`. This needs far more memory (the 3B model's fp32 optimizer state alone is about 36GB).

## 4. Output

Checkpoints go to `outputs/sft/checkpoint-*`, the final adapter to `outputs/sft/final_adapter`. Load it with:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "outputs/sft/final_adapter")
```
