# LoRA SFT for Qwen2.5-3B-Instruct

Fine-tunes `Qwen2.5-3B-Instruct` with LoRA on coding and persona conversations, using [Miles](https://github.com/radixark/miles) (slime with LoRA support).

```
configs/common.yaml    paths, system prompt, LoRA rank and alpha
configs/sft.yaml       data path, save dir, epochs, batch size, learning rate
prepare_sft_data.py    jsonl -> data/sft.parquet
run_sft_lora.py        starts training
```

Scripts take no arguments. Edit the YAML files and the paths at the top of `prepare_sft_data.py`.

## 1. Setup

Training runs inside the `radixark/miles` Docker image (it ships the patched SGLang and Megatron that Miles needs).

```bash
git clone -b sft https://github.com/essacheez/Posttraining-coding-rocky.git qwen_posttrain
cd qwen_posttrain
mkdir -p ~/models ~/outputs

docker run --rm -it --gpus all --ipc=host --shm-size=32g --network=host \
  -v $PWD:/root/qwen_posttrain -v $HOME/models:/root/models -v $HOME/outputs:/root/outputs \
  -w /root/qwen_posttrain radixark/miles:latest /bin/bash
```

Inside the container:

```bash
cd /root/miles && git fetch --all && git checkout 50ad28b47c8b7cd1f6bfed7767bb505a9fc8d9a9 && pip install -e . --no-deps
cd /root/qwen_posttrain && uv pip install --system -r pyproject.toml
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/models/Qwen2.5-3B-Instruct
```

The paths in the YAML files match this layout.

## 2. Data

One JSON object per line, either format:

```json
{"prompt": "Write a function that reverses a string.", "response": "```python\ndef reverse(s):\n    return s[::-1]\n```"}
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Kit, ..."}]}
```

The last turn must be the assistant. Rows without a system message get the `system_prompt` from `configs/common.yaml`.

Set `INPUT_FILES` at the top of `prepare_sft_data.py` to your files, then:

```bash
python prepare_sft_data.py
```

## 3. Train

```bash
python run_sft_lora.py
```

Settings are in `configs/sft.yaml`. Start with a short run (`num_epoch: 1`, `save_interval: 5`) to check everything works.

- **More or fewer GPUs**: change `num_gpus`. Training is data parallel, so `global_batch_size` must be divisible by it.
- **Out of memory**: lower `max_tokens_per_gpu`.
- **Full fine-tuning instead of LoRA**: remove the `--lora-rank ... --target-modules all-linear` line from `model_args` in `run_sft_lora.py` and set `lr` to about `1.0e-5`. This needs about 48GB of GPU memory for optimizer state (not 24GB cards), and checkpoints become Megatron format; convert with `python /root/miles/tools/convert_torch_dist_to_hf.py --input-dir <iter dir> --output-dir <out> --origin-hf-dir /root/models/Qwen2.5-3B-Instruct`.

## 4. Output

Adapters are saved to `save_dir/iter_XXXXXXX/adapter/`. Load one with `peft`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "/root/outputs/qwen_posttrain/sft/iter_0000150/adapter")
```
