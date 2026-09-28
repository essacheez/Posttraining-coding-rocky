"""GRPO with verifiable rewards on top of the RLAIF model, keeping the persona with the RLAIF rewards.
RLVR_MODE=verifier is the correctness only attempt; RLVR_MODE=multi adds the judge, marker and format rewards."""

import os
import sys

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

RLVR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(RLVR_DIR)
sys.path.insert(0, REPO_ROOT)

# imported, not copied, so the persona reward here is the same instrument RLAIF trained against
from RLAIF.training.train_rlaif import OPENAI_API_KEY, format_reward, marker_penalty, run_judge, style_reward
from RLVR.training.merge_adapter import RLAIF_MERGED_DIR
from RLVR.training.verifier import verify_all

RLVR_MODE = os.environ.get("RLVR_MODE", "multi")
ATTEMPT = os.environ.get("ATTEMPT", f"attempt_{RLVR_MODE}")
DATA_FILE = os.path.join(RLVR_DIR, "data", "rlvr_dataset.jsonl")
OUTPUT_DIR = os.path.join(RLVR_DIR, "model", ATTEMPT)

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

NUM_GENERATIONS = 8
MAX_STEPS = int(os.environ.get("MAX_STEPS", 150))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 1e-5))
LR_SCHEDULER = "cosine"
WARMUP_STEPS = 5
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
MAX_COMPLETION_LENGTH = 512
BETA = float(os.environ.get("BETA", 0.04))
LOGGING_STEPS = 1
SAVE_STEPS = 25
SEED = int(os.environ.get("SEED", 0))
VLLM_GPU_MEMORY_UTILIZATION = 0.3

# verifier, judge task, judge style, marker penalty, format; verifier is 0 to 1, so 5x puts it on the judge's 0 to 5 scale
DEFAULT_MULTI_WEIGHTS = "5.0,1.0,0.6,1.0,1.0"
MULTI_REWARD_WEIGHTS = [float(weight) for weight in os.environ.get("REWARD_WEIGHTS", DEFAULT_MULTI_WEIGHTS).split(",")]

WANDB_PROJECT = "qwen-posttrain-trl"
RUN_NAME = f"qwen2.5-3b-rlvr-{ATTEMPT}"


def verifier_reward(prompts, completions, kind, tests, setup, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    scores = verify_all(responses, tests, setup)
    rewards = []
    for score, row_kind in zip(scores, kind):
        rewards.append(score if row_kind == "stem" else None)
    return rewards


def judge_task_reward(prompts, completions, reference, kind, **kwargs):
    # runs the judge on every completion and fills the cache style_reward reads, so it must come first
    verdicts = run_judge(prompts, completions, reference)
    rewards = []
    for verdict, row_kind in zip(verdicts, kind):
        rewards.append(verdict["task"] if verdict and row_kind == "persona" else None)
    return rewards


def rewards_for_mode():
    if RLVR_MODE == "verifier":
        return [verifier_reward], [1.0]
    if RLVR_MODE == "multi":
        reward_funcs = [verifier_reward, judge_task_reward, style_reward, marker_penalty, format_reward]
        if len(MULTI_REWARD_WEIGHTS) != len(reward_funcs):
            raise SystemExit(f"REWARD_WEIGHTS needs {len(reward_funcs)} values, got {len(MULTI_REWARD_WEIGHTS)}")
        return reward_funcs, MULTI_REWARD_WEIGHTS
    raise SystemExit(f"RLVR_MODE must be 'verifier' or 'multi', got {RLVR_MODE!r}")


def main():
    reward_funcs, reward_weights = rewards_for_mode()
    if RLVR_MODE == "multi" and OPENAI_API_KEY == "API_KEY":
        raise SystemExit("set OPENAI_API_KEY in .env, the multi objective reward calls the RLAIF judge")
    if not os.path.isdir(RLAIF_MERGED_DIR):
        raise SystemExit(f"{RLAIF_MERGED_DIR} missing, run RLVR/training/merge_adapter.py first")

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.rename_column("completion", "reference")
    if RLVR_MODE == "verifier":
        dataset = dataset.filter(lambda row: row["kind"] == "stem")
    print(f"{ATTEMPT}: {len(dataset)} prompts, rewards {[func.__name__ for func in reward_funcs]}, weights {reward_weights}")

    tokenizer = AutoTokenizer.from_pretrained(RLAIF_MERGED_DIR)
    merged_model = AutoModelForCausalLM.from_pretrained(RLAIF_MERGED_DIR, dtype="bfloat16")
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(merged_model, lora_config)

    training_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA,
        reward_weights=reward_weights,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="no",
        report_to="wandb",
        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "RLVR_model"))


if __name__ == "__main__":
    main()
