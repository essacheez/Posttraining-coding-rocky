# LoRA SFT for Qwen2.5-3B-Instruct

Supervised fine-tuning of `Qwen2.5-3B-Instruct` with LoRA on coding and persona conversations, using [Miles](https://github.com/radixark/miles) (a fork of slime with LoRA support). Runs on 2 x 24GB GPUs or a single larger GPU.

## Structure

```
configs/common.yaml    system prompt, paths to miles / Megatron / model, LoRA rank and alpha, fixed Megatron flags
configs/sft.yaml       data path, save dir, epochs, batch size, learning rate
prepare_sft_data.py    your jsonl files -> data/sft.parquet
run_sft_lora.py        launches training
pyproject.toml         python dependencies for the data script
```

No script takes command line arguments. Edit the YAML files and the paths at the top of `prepare_sft_data.py`.

## 1. Set up the environment

Training runs inside the `radixark/miles` Docker image. It bundles PyTorch, patched SGLang, Megatron-LM, Megatron-Bridge, TransformerEngine and Miles itself; these cannot be pip installed, so Docker is the only supported route.

On the GPU machine, with Docker and the NVIDIA container toolkit installed:

```bash
# 1. get the code
git clone -b sft https://github.com/essacheez/Posttraining-coding-rocky.git qwen_posttrain
cd qwen_posttrain

# 2. pull the image and start a container with the code, models and outputs mounted
docker pull radixark/miles:latest
mkdir -p ~/models ~/outputs
docker run --rm -it --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 --network=host \
  -v $PWD:/root/qwen_posttrain -v $HOME/models:/root/models -v $HOME/outputs:/root/outputs \
  -w /root/qwen_posttrain radixark/miles:latest /bin/bash
```

Everything below runs inside the container.

```bash
# 3. pin Miles to the commit this code was written against
cd /root/miles && git fetch --all && git checkout 50ad28b47c8b7cd1f6bfed7767bb505a9fc8d9a9 && pip install -e . --no-deps

# 4. install the small dependencies of the data script (pyarrow, pyyaml, ...)
cd /root/qwen_posttrain && uv pip install --system -r pyproject.toml

# 5. download the base model
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/models/Qwen2.5-3B-Instruct

# 6. check
python -c "import miles; print('ok')" && nvidia-smi
```

The paths in `configs/common.yaml` (`/root/miles`, `/root/Megatron-LM`, `/root/models/Qwen2.5-3B-Instruct`) and `configs/sft.yaml` (`/root/qwen_posttrain/data/sft.parquet`, `/root/outputs/qwen_posttrain/sft`) match this layout. Change them if you mount things elsewhere.

To run only the data script on a laptop without GPUs, `uv sync` in the repo creates a local environment (`uv run python prepare_sft_data.py`).

## 2. Prepare the data

Write your conversations as jsonl files, one JSON object per line. Both formats are accepted and can be mixed:

```json
{"prompt": "Write a function that reverses a string.", "response": "Slicing does it in one line.\n\n```python\ndef reverse(s):\n    return s[::-1]\n```"}
{"messages": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I'm Kit, ..."}]}
```

- `messages` rows can have several user/assistant turns; the last turn must be `assistant`. Loss is computed on assistant turns only.
- A row with its own `system` message keeps it. Rows without one get the `system_prompt` from `configs/common.yaml`.
- Coding and persona data go through the same script; put them in separate files or one file, as you like.

Point `INPUT_FILES` at the top of `prepare_sft_data.py` to your files (default `data/raw/coding.jsonl` and `data/raw/persona.jsonl`), then:

```bash
python prepare_sft_data.py
```

This shuffles all rows and writes `data/sft.parquet` with a single `messages` column, which is what `configs/sft.yaml` points to.

## 3. Run SFT

```bash
python run_sft_lora.py
```

The script reads `configs/common.yaml` and `configs/sft.yaml`, starts ray, and submits the Miles training job. Logs stream to the terminal.

Settings in `configs/sft.yaml`:

| key | meaning |
|---|---|
| `data` | the parquet from step 2 |
| `save_dir`, `save_interval` | where and how often adapters are saved |
| `num_gpus` | 2 for 2 x 24GB (data parallel), 1 for a single GPU |
| `num_epoch`, `global_batch_size` | training length and batch size |
| `max_tokens_per_gpu` | lower this if you run out of memory |
| `lr`, `min_lr`, `lr_warmup_fraction`, `weight_decay`, `adam_beta1`, `adam_beta2` | optimizer |

LoRA rank and alpha are in `configs/common.yaml`. Do a short smoke run first (`num_epoch: 1`, `save_interval: 5` on a small file) before the full run.

### Different number of GPUs

Only `num_gpus` in `configs/sft.yaml` changes. SFT runs data parallel (tensor parallel 1), so N GPUs means N copies of the model each taking a slice of every batch. `global_batch_size` must be divisible by `num_gpus`. Per-GPU memory is unchanged, so a single 48GB card works the same as one 24GB card, just with room to raise `max_tokens_per_gpu` (for example 16384) to pack more tokens per step.

### Full fine-tuning instead of LoRA

LoRA is enabled by `--lora-rank` in `run_sft_lora.py`. To train all weights, delete the LoRA line from `model_args`:

```python
model_args = (
    f"--hf-checkpoint {COMMON_ARGUMENTS['model_dir']} --megatron-to-hf-mode bridge "
)
```

and lower `lr` in `configs/sft.yaml` to about `1.0e-5` (LoRA tolerates a higher rate than full weights do).

Two things change with full fine-tuning:

- **Memory.** Megatron keeps fp32 master weights and Adam moments, about 16 bytes per parameter, so roughly 48GB for the 3B model before activations. Data parallel does not shard that. It does not fit on 24GB cards; use 2 x 48GB or larger, or add `--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer ` to `optimizer_args` to keep optimizer state in host RAM (that is what Miles' own SFT recipe does for large models). This path is not tested here.
- **Output format.** Without LoRA, checkpoints are Megatron `torch_dist` directories, not HF adapters. Convert one to an HF model with Miles' tool inside the container:

  ```bash
  cd /root/miles && python tools/convert_torch_dist_to_hf.py \
    --input-dir /root/outputs/qwen_posttrain/sft/iter_0000150 \
    --output-dir /root/outputs/qwen_posttrain/sft_hf \
    --origin-hf-dir /root/models/Qwen2.5-3B-Instruct
  ```

## 4. Output

Each `save_dir/iter_XXXXXXX/adapter/` contains Megatron-native shards (for Miles to resume from) and an HF PEFT `adapter_model.bin` + `adapter_config.json` you can load with `peft` on top of `Qwen2.5-3B-Instruct`:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "/root/outputs/qwen_posttrain/sft/iter_0000150/adapter")
```
