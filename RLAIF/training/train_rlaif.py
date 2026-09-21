import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from openai import OpenAI
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

RLAIF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(RLAIF_DIR)
sys.path.insert(0, REPO_ROOT)

from RLAIF.training.merge_adapter import MERGED_MODEL_DIR

DATA_FILE = os.path.join(RLAIF_DIR, "data", "grpo_dataset.jsonl")
OUTPUT_DIR = os.path.join(RLAIF_DIR, "model")

def load_dotenv(path):
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(os.path.join(REPO_ROOT, ".env"))

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "API_KEY")
JUDGE_MODEL = "gpt-5.4-nano"
JUDGE_MAX_WORKERS = 16

JUDGE_PROMPT_FILE = os.path.join(RLAIF_DIR, "training", "judge.txt")
with open(JUDGE_PROMPT_FILE, encoding="utf-8") as judge_prompt_in:
    JUDGE_PROMPT = judge_prompt_in.read()

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

NUM_GENERATIONS = 8
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-5
LR_SCHEDULER = "constant"
WARMUP_STEPS = 5
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
MAX_COMPLETION_LENGTH = 1024
BETA = 0.004
LOGGING_STEPS = 1
SAVE_STEPS = 15
VAL_SPLIT = 0.10
SEED = 0

VLLM_GPU_MEMORY_UTILIZATION = 0.3

WANDB_PROJECT = "qwen-posttrain-trl"
RUN_NAME = "qwen2.5-3b-grpo-lora"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

JUDGE_FAILURE_RATE_THRESHOLD = 0.3
JUDGE_MIN_BATCH_FOR_THRESHOLD = 8


def parse_score(raw):
    parsed = json.loads(raw)
    return parsed["task_score"] + parsed["rocky_style_score"]


def judge_one(question, reference, response):
    prompt = JUDGE_PROMPT.replace("{QUESTION}", question).replace("{REFERENCE_ANSWER}", reference).replace("{MODEL_RESPONSE}", response)
    try:
        completion = openai_client.chat.completions.create(model=JUDGE_MODEL, temperature=0.2, messages=[{"role": "user", "content": prompt}])
        raw = completion.choices[0].message.content
    except Exception as e:
        print(f"[judge] API call failed, skipping this completion: {e}", file=sys.stderr)
        return None

    try:
        return parse_score(raw)
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"[judge] malformed judge output, skipping this completion: {raw!r} ({e})", file=sys.stderr)
        return None


def persona_reward(prompts, completions, reference, **kwargs):
    questions = [prompt[-1]["content"] for prompt in prompts]
    responses = [completion[0]["content"] for completion in completions]
    references = [ref[-1]["content"] for ref in reference]
    with ThreadPoolExecutor(max_workers=JUDGE_MAX_WORKERS) as pool:
        scores = list(pool.map(judge_one, questions, references, responses))

    failed = sum(1 for s in scores if s is None)
    if len(scores) >= JUDGE_MIN_BATCH_FOR_THRESHOLD and failed / len(scores) > JUDGE_FAILURE_RATE_THRESHOLD:
        raise RuntimeError(
            f"persona_reward: {failed}/{len(scores)} judge calls failed in this batch "
            f"(> {JUDGE_FAILURE_RATE_THRESHOLD:.0%} threshold) - judge appears broken, aborting run"
        )
    return scores


def main():
    if OPENAI_API_KEY == "API_KEY":
        raise SystemExit("set OPENAI_API_KEY at the top of train_rlaif.py before running")

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.rename_column("completion", "reference")
    dataset = dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR)
    merged_model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_DIR, dtype="bfloat16")

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
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=int(os.environ.get("SMOKE_TEST_STEPS", 0)) or -1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=SAVE_STEPS,
        report_to="wandb",
        seed=SEED,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        reward_funcs=persona_reward,
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_adapter"))


if __name__ == "__main__":
    main()
