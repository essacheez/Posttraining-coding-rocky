"""Phase 1: LoRA SFT of Qwen2.5-3B-Instruct on coding + persona conversations. Settings in configs/sft.yaml.

Runs on 2 x 24GB GPUs as data parallel (TP1, DP2). No SGLang engine is started.
"""

import os
import miles.utils.external_utils.command_utils as command_utils
import yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_DIR, "configs", "common.yaml"), encoding="utf-8") as config_file:
    COMMON_ARGUMENTS = yaml.safe_load(config_file)
with open(os.path.join(PROJECT_DIR, "configs", "sft.yaml"), encoding="utf-8") as config_file:
    SFT_ARGUMENTS = yaml.safe_load(config_file)


def main():
    model_args = (
        f"--hf-checkpoint {COMMON_ARGUMENTS['model_dir']} --megatron-to-hf-mode bridge "
        f"--lora-rank {COMMON_ARGUMENTS['lora_rank']} --lora-alpha {COMMON_ARGUMENTS['lora_alpha']} "
        "--lora-dropout 0.0 --target-modules all-linear "
    )

    sft_args = (
        "--rollout-function-path miles.rollout.sft_rollout.generate_rollout "
        "--loss-type sft_loss --loss-mask-type qwen --input-key messages "
        "--disable-compute-advantages-and-returns --debug-train-only --rollout-shuffle "
        f"--prompt-data {SFT_ARGUMENTS['data']} "
        f"--save {SFT_ARGUMENTS['save_dir']} --save-interval {SFT_ARGUMENTS['save_interval']} "
        f"--num-epoch {SFT_ARGUMENTS['num_epoch']} "
        f"--rollout-batch-size {SFT_ARGUMENTS['global_batch_size']} "
        f"--global-batch-size {SFT_ARGUMENTS['global_batch_size']} "
        f"--max-tokens-per-gpu {SFT_ARGUMENTS['max_tokens_per_gpu']} "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {SFT_ARGUMENTS['num_gpus']} "
        f"--num-gpus-per-node {SFT_ARGUMENTS['num_gpus']} "
    )

    optimizer_args = (
        "--optimizer adam --lr-decay-style cosine "
        f"--lr {SFT_ARGUMENTS['lr']} --min-lr {SFT_ARGUMENTS['min_lr']} "
        f"--lr-warmup-fraction {SFT_ARGUMENTS['lr_warmup_fraction']} "
        f"--weight-decay {SFT_ARGUMENTS['weight_decay']} "
        f"--adam-beta1 {SFT_ARGUMENTS['adam_beta1']} --adam-beta2 {SFT_ARGUMENTS['adam_beta2']} "
    )

    fixed_args = " ".join(COMMON_ARGUMENTS["fixed_args"])

    command_utils.execute_train(
        train_args=model_args + sft_args + optimizer_args + fixed_args,
        num_gpus_per_node=SFT_ARGUMENTS["num_gpus"],
        megatron_model_type=COMMON_ARGUMENTS["megatron_model_type"],
        train_script=os.path.join(COMMON_ARGUMENTS["miles_dir"], "train_async.py"),
        megatron_path=COMMON_ARGUMENTS["megatron_dir"],
        extra_env_vars={"PYTHONPATH": f"{COMMON_ARGUMENTS['miles_dir']}:{PROJECT_DIR}"},
    )


if __name__ == "__main__":
    main()
