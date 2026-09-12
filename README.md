# LoRA SFT for Qwen2.5-3B-Instruct

Supervised fine-tuning of `Qwen2.5-3B-Instruct` with LoRA on coding and persona conversations, using [Miles](https://github.com/radixark/miles) (a fork of slime with LoRA support). Runs on 2 x 24GB GPUs or a single larger GPU.

## Structure

```
configs/common.yaml    system prompt, paths to miles / Megatron / model, LoRA rank and alpha, fixed Megatron flags
configs/sft.yaml       data path, save dir, epochs, batch size, learning rate
prepare_sft_data.py    your jsonl files -> data/sft.parquet
run_sft_lora.py        launches training
pyproject.toml         uv project for the data script
```

No script takes command line arguments. Edit the YAML files and the paths at the top of `prepare_sft_data.py`.

## Data

Put your conversations in jsonl files, one row per conversation, in either format:

```json
{"prompt": "Write a function that ...", "response": "Here is ..."}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Rows without a system message get the `system_prompt` from `configs/common.yaml`. List the files in `INPUT_FILES` at the top of `prepare_sft_data.py`, then:

```
uv sync
uv run python prepare_sft_data.py
```

This writes `data/sft.parquet`, which is what `configs/sft.yaml` points to.

## Environment

Training needs the `radixark/miles` Docker image (it bundles patched SGLang, Megatron-LM, Megatron-Bridge and TransformerEngine, which cannot be pip installed).

```
docker run --rm -it --gpus all --ipc=host --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 --network=host \
  -v $PWD:/root/qwen_posttrain -v $HOME/models:/root/models -v $HOME/outputs:/root/outputs \
  -w /root/qwen_posttrain radixark/miles:latest /bin/bash

# inside the container
cd /root/miles && git fetch --all && git checkout 50ad28b47c8b7cd1f6bfed7767bb505a9fc8d9a9 && pip install -e . --no-deps
cd /root/qwen_posttrain && uv pip install --system -r pyproject.toml
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /root/models/Qwen2.5-3B-Instruct
```

The paths in `configs/common.yaml` and `configs/sft.yaml` match this layout. Change them if you mount things elsewhere.

## Train

```
python run_sft_lora.py
```

Settings in `configs/sft.yaml`:

| key | meaning |
|---|---|
| `data` | the parquet from `prepare_sft_data.py` |
| `save_dir`, `save_interval` | where and how often adapters are saved |
| `num_gpus` | 2 for 2 x 24GB (data parallel), 1 for a single GPU |
| `num_epoch`, `global_batch_size` | training length and batch size |
| `max_tokens_per_gpu` | lower this if you run out of memory |
| `lr`, `min_lr`, `lr_warmup_fraction`, `weight_decay`, `adam_beta1`, `adam_beta2` | optimizer |

Do a short smoke run first (`num_epoch: 1`, `save_interval: 5` on a small file).

## Output

Each `save_dir/iter_XXXXXXX/adapter/` contains Megatron-native shards (for Miles to resume from) and an HF PEFT `adapter_model.bin` + `adapter_config.json` you can load with `peft` on top of `Qwen2.5-3B-Instruct`.
