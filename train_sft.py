import os
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
DATA_FILE = os.path.join(PROJECT_DIR, "data", "sft.jsonl")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "sft")

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
LR_SCHEDULER = "cosine"
WARMUP_STEPS = 20
WEIGHT_DECAY = 0.0
PER_DEVICE_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8    # effective batch = 4 x 8 x num_gpus
MAX_LENGTH = 4096
LOSS_TYPE = "nll"    # trl's default "chunked_nll" errors out on a LoRA-wrapped model in this trl version
LOGGING_STEPS = 10
SAVE_STEPS = 50
EVAL_STEPS = 50
VAL_SPLIT = 0.05
SEED = 0

WANDB_PROJECT = "qwen-posttrain-trl"
RUN_NAME = "qwen2.5-3b-sft-lora"


def main():
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )

    training_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_length=MAX_LENGTH,
        loss_type=LOSS_TYPE,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        model_init_kwargs={"dtype": "bfloat16"},
        report_to="wandb",
    )

    trainer = SFTTrainer(
        model=MODEL_ID,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))


if __name__ == "__main__":
    main()
