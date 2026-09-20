"""GRPO against the persona judge, starting from the SFT checkpoint.

Mirrors train_sft.py: same base model, same LoRA shape, same conventions, so
checkpoint 2 is a continuation of checkpoint 1 rather than a second project.

    uv run python rlaif/train_grpo.py                      # gated reward
    REWARD_MODE=sum uv run python rlaif/train_grpo.py       # plain 5-25 total

The model loaded here is the *merged* SFT checkpoint, and the LoRA adapter this
trains is a fresh one on top of it. That is what makes the KL reference right:
TRL computes reference logprobs with the adapter disabled, which is the SFT
policy, so beta measures drift from where checkpoint 1 finished rather than from
the base model.
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# run as a script, sys.path[0] is rlaif/, which leaves the repo-root modules out
sys.path.insert(0, PROJECT_DIR)

from rlaif.reward import MAX_COMPLETION_LENGTH, REWARD_MODE, persona_reward

MODEL_DIR = os.path.join(PROJECT_DIR, "outputs", "sft", "merged_hf")
DATA_FILE = os.path.join(PROJECT_DIR, "data", "rlaif_prompts.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "rlaif", f"grpo_{REWARD_MODE}")

# same LoRA shape as checkpoint 1, so the two stages are comparable
LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

NUM_GENERATIONS = 8          # the group GRPO takes its advantage over
PROMPTS_PER_STEP = 8
GRAD_ACCUM = 8
LEARNING_RATE = 1e-6
NUM_EPOCHS = 1
TEMPERATURE = 1.0
# the persona already sits at the human ceiling (EVAL_RESULTS.md section 4), so
# the risk being managed here is drift away from it, not underfitting
BETA = 0.001
# prompt + completion; TRL 1.13 has no separate max_prompt_length
VLLM_MAX_MODEL_LENGTH = 4096
SAVE_STEPS = 10
LOGGING_STEPS = 1
SEED = 0

WANDB_PROJECT = "qwen-posttrain-trl"
RUN_NAME = f"qwen2.5-3b-grpo-{REWARD_MODE}"


def main():
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set; the judge reward needs it")
    if not os.path.isdir(MODEL_DIR):
        raise SystemExit(f"{MODEL_DIR} not found, run: uv run python rlaif/merge_lora.py")
    if not os.path.isfile(DATA_FILE):
        raise SystemExit(f"{DATA_FILE} not found, run: uv run python rlaif/prepare_prompts.py")

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    train_dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    training_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        num_generations=NUM_GENERATIONS,
        per_device_train_batch_size=NUM_GENERATIONS,
        gradient_accumulation_steps=GRAD_ACCUM,
        generation_batch_size=NUM_GENERATIONS * PROMPTS_PER_STEP,
        temperature=TEMPERATURE,
        max_completion_length=MAX_COMPLETION_LENGTH,
        vllm_max_model_length=VLLM_MAX_MODEL_LENGTH,
        # vLLM in-process, so rollouts do not run through the training forward pass
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.3,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        log_completions=True,
        model_init_kwargs={"dtype": "bfloat16"},
        report_to="wandb",
        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=MODEL_DIR,
        reward_funcs=persona_reward,
        args=training_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))


if __name__ == "__main__":
    main()
