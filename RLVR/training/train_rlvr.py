import os
import re
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from verifier import verify

RLVR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(RLVR_DIR)

DATA_FILE = os.path.join(RLVR_DIR, "data", "rlvr_dataset.jsonl")  # rows with a "prompt" chat and a "test_cases" column
OUTPUT_DIR = os.path.join(RLVR_DIR, "model")
MODEL_DIR = os.path.join(RLVR_DIR, "model", "rlaif_merged")  # SFT + RLAIF merged into one model we train a new adapter on top


LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

NUM_GENERATIONS = 8
MAX_COMPLETION_LENGTH =  3072
VLLM_MAX_MODEL_LENGTH = 4940  # longest prompt is 1868 tokens, plus the completion
NUM_TRAIN_EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 1e-5
LR_SCHEDULER = "constant"
WARMUP_STEPS = 5
BETA = 0.01
LOGGING_STEPS = 1
SAVE_STEPS = 25
SEED = 0
VLLM_GPU_MEMORY_UTILIZATION = 0.3
NUM_VERIFY_WORKERS = 6  # CPUs per GPU process in the Slurm job

WANDB_PROJECT = "qwen-posttrain-trl"
RUN_NAME = "qwen2.5-3b-rlvr-lora"


CODE_BLOCK_PATTERN = re.compile(r"(?<!`)(`{2,})(?!`)([^\n]*)\n(.*?)(?<!`)\1(?!`)", re.DOTALL)  # a full run of 2+ backticks, closed by a run of the same length
FENCE_TAG_PATTERN = re.compile(r"[\w .+#-]*")  # python, py, python3, "python code", or nothing


def extract_code(response):
    """Return the code from the single fenced block, or None if there is not exactly one block."""
    blocks = CODE_BLOCK_PATTERN.findall(response)
    if len(blocks) != 1:
        return None

    fence, fence_line, code = blocks[0]
    if not FENCE_TAG_PATTERN.fullmatch(fence_line):  # code written on the fence line itself, e.g. ```def f(x):
        code = fence_line + "\n" + code

    return code


def score_completion(completion, tests):
    code = extract_code(completion[0]["content"])
    if code is None:
        return 0.0
    return verify(code, tests)


def code_reward(completions, test_cases, **kwargs):
    with ThreadPoolExecutor(max_workers=NUM_VERIFY_WORKERS) as pool:
        return list(pool.map(score_completion, completions, test_cases))


def main():
    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    train_dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    merged_model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype="bfloat16")

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
        seed=SEED,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        shuffle_dataset=False,  # the dataset is in curriculum order, easy to very hard
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        vllm_max_model_length=VLLM_MAX_MODEL_LENGTH,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="no",
        report_to="wandb",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[code_reward],
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "RLVR_model"))


if __name__ == "__main__":
    main()
