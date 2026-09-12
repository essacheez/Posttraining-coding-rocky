"""Phase 1: LoRA SFT of Qwen2.5-3B-Instruct on coding + persona conversations."""

import os

import miles.utils.external_utils.command_utils as command_utils
import yaml

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(PROJECT_DIR, "configs", "common.yaml"), encoding="utf-8") as config_file:
    COMMON_ARGUMENTS = yaml.safe_load(config_file)
with open(os.path.join(PROJECT_DIR, "configs", "sft.yaml"), encoding="utf-8") as config_file:
    SFT_ARGUMENTS = yaml.safe_load(config_file)

def yaml_args_to_miles_flags(arguments):
    flags = []
    for key, value in arguments.items():
        flag = "--" + key.replace("_", "-")
        if value is True:
            flags.append(flag)
        elif value is not None and value is not False:
            flags.append(f"{flag} {value}")
    return " ".join(flags)


def main():
    launcher = COMMON_ARGUMENTS["launcher"]
    train_args = {**COMMON_ARGUMENTS["train_args"], **SFT_ARGUMENTS["train_args"]}

    command_utils.execute_train(
        train_args=yaml_args_to_miles_flags(train_args),
        num_gpus_per_node=train_args["num_gpus_per_node"],
        megatron_model_type=launcher["megatron_model_type"],
        train_script=os.path.join(launcher["miles_dir"], "train_async.py"),
        megatron_path=launcher["megatron_dir"],
        extra_env_vars={"PYTHONPATH": f"{launcher['miles_dir']}:{PROJECT_DIR}"},
    )


if __name__ == "__main__":
    main()
