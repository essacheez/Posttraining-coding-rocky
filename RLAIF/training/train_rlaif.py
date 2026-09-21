import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from datasets import load_dataset
from openai import OpenAI
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

RLAIF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(RLAIF_DIR)
sys.path.insert(0, REPO_ROOT)

from RLAIF.training.merge_adapter import MERGED_MODEL_DIR

DATA_FILE = os.path.join(RLAIF_DIR, "data", "grpo_dataset.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(RLAIF_DIR, "model"))
# set to an adapter dir to continue GRPO from an earlier checkpoint instead of a fresh LoRA
INIT_ADAPTER = os.environ.get("INIT_ADAPTER", "")


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
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_WORKERS = 32

JUDGE_PROMPT_FILE = os.path.join(RLAIF_DIR, "training", "judge.txt")
with open(JUDGE_PROMPT_FILE, encoding="utf-8") as judge_prompt_in:
    JUDGE_PROMPT = judge_prompt_in.read()

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

NUM_GENERATIONS = 8
NUM_TRAIN_EPOCHS = 1
MAX_STEPS = int(os.environ.get("SMOKE_TEST_STEPS", 0)) or int(os.environ.get("MAX_STEPS", 0)) or -1
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 2e-6))
LR_SCHEDULER = os.environ.get("LR_SCHEDULER", "cosine")
WARMUP_STEPS = 5
PER_DEVICE_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 8
MAX_COMPLETION_LENGTH = 512
BETA = 0.04
LOGGING_STEPS = 1
SAVE_STEPS = int(os.environ.get("SAVE_STEPS", 15))
VAL_SPLIT = 0.05
SEED = int(os.environ.get("SEED", 0))

# task, style, marker penalty, format (code block count + length)
REWARD_WEIGHTS = [1.0, 0.6, 1.0, 1.0]

# per 1k word rates in the Rocky reference corpus, from the persona eval
REFERENCE_MARKER_RATES = {"listen": 7.0, "you": 13.0, "yes": 6.0, "good": 15.0}
MARKER_TOLERANCE = 1.5
MARKER_PENALTY_SCALE = 0.5
MARKER_PENALTY_FLOOR = -2.0
LISTEN_OPENER_PENALTY = 0.5
LENGTH_TOLERANCE = 1.5
LENGTH_PENALTY_FLOOR = -1.0

VLLM_GPU_MEMORY_UTILIZATION = 0.3

WANDB_PROJECT = "qwen-posttrain-trl"
RUN_NAME = "qwen2.5-3b-grpo-lora"

openai_client = OpenAI(api_key=OPENAI_API_KEY)

JUDGE_FAILURE_RATE_THRESHOLD = 0.3
JUDGE_MIN_BATCH_FOR_THRESHOLD = 8

# one judge call per completion, shared by task_reward and style_reward within a batch
JUDGE_CACHE = {}

FENCE_PATTERN = re.compile(r"```(?:[a-zA-Z]*)\n(.*?)```", re.DOTALL)


def judge_one(question, reference, response):
    prompt = JUDGE_PROMPT.replace("{QUESTION}", question).replace("{REFERENCE_ANSWER}", reference).replace("{MODEL_RESPONSE}", response)
    try:
        completion = openai_client.chat.completions.create(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = completion.choices[0].message.content
    except Exception as error:
        print(f"[judge] API call failed, skipping this completion: {error}", file=sys.stderr)
        return None

    try:
        parsed = json.loads(raw)
        return {"task": float(parsed["task_score"]), "style": float(parsed["rocky_style_score"])}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        print(f"[judge] malformed judge output, skipping this completion: {raw!r} ({error})", file=sys.stderr)
        return None


def unpack(prompts, completions, reference):
    questions = [prompt[-1]["content"] for prompt in prompts]
    responses = [completion[0]["content"] for completion in completions]
    references = [ref[-1]["content"] for ref in reference]
    return questions, responses, references


def cache_key(question, response):
    return question + "\n<<<SEP>>>\n" + response


def run_judge(prompts, completions, reference):
    questions, responses, references = unpack(prompts, completions, reference)
    with ThreadPoolExecutor(max_workers=JUDGE_MAX_WORKERS) as pool:
        verdicts = list(pool.map(judge_one, questions, references, responses))

    failed = sum(1 for verdict in verdicts if verdict is None)
    if len(verdicts) >= JUDGE_MIN_BATCH_FOR_THRESHOLD and failed / len(verdicts) > JUDGE_FAILURE_RATE_THRESHOLD:
        raise RuntimeError(
            f"judge: {failed}/{len(verdicts)} judge calls failed in this batch "
            f"(> {JUDGE_FAILURE_RATE_THRESHOLD:.0%} threshold) - judge appears broken, aborting run"
        )

    JUDGE_CACHE.clear()
    for question, response, verdict in zip(questions, responses, verdicts):
        JUDGE_CACHE[cache_key(question, response)] = verdict
    return verdicts


def task_reward(prompts, completions, reference, **kwargs):
    verdicts = run_judge(prompts, completions, reference)
    return [verdict["task"] if verdict else None for verdict in verdicts]


def style_reward(prompts, completions, reference, **kwargs):
    questions, responses, references = unpack(prompts, completions, reference)
    verdicts = [JUDGE_CACHE.get(cache_key(question, response)) for question, response in zip(questions, responses)]
    return [verdict["style"] if verdict else None for verdict in verdicts]


def marker_rates_per_1k(text):
    words = re.findall(r"[a-z']+", text.lower())
    total = max(len(words), 1)
    rates = {}
    for marker in REFERENCE_MARKER_RATES:
        rates[marker] = 1000.0 * words.count(marker) / total
    return rates


def marker_penalty_one(response):
    rates = marker_rates_per_1k(response)
    excess = 0.0
    for marker, reference_rate in REFERENCE_MARKER_RATES.items():
        excess += max(0.0, rates[marker] - MARKER_TOLERANCE * reference_rate) / reference_rate
    penalty = -MARKER_PENALTY_SCALE * excess
    if response.lstrip().lower().startswith("listen"):
        penalty -= LISTEN_OPENER_PENALTY
    return max(MARKER_PENALTY_FLOOR, penalty)


def marker_penalty(prompts, completions, reference, **kwargs):
    questions, responses, references = unpack(prompts, completions, reference)
    return [marker_penalty_one(response) for response in responses]


def is_code_prompt(question, reference_answer):
    return "```" in reference_answer or "python" in question.lower()


def code_format_score(response):
    blocks = FENCE_PATTERN.findall(response)
    outside = FENCE_PATTERN.sub("", response)
    if re.search(r"^\s*def \w+\(", outside, re.MULTILINE):
        return -1.0
    if len(blocks) == 1:
        return 1.0
    if len(blocks) == 2:
        return 0.0
    return -1.0


def length_penalty_one(response, reference_answer):
    response_words = len(response.split())
    reference_words = max(len(reference_answer.split()), 20)
    ratio = response_words / reference_words
    if ratio <= LENGTH_TOLERANCE:
        return 0.0
    return max(LENGTH_PENALTY_FLOOR, -(ratio - LENGTH_TOLERANCE))


def format_reward(prompts, completions, reference, **kwargs):
    questions, responses, references = unpack(prompts, completions, reference)
    scores = []
    for question, response, reference_answer in zip(questions, responses, references):
        score = length_penalty_one(response, reference_answer)
        if is_code_prompt(question, reference_answer):
            score += code_format_score(response)
        scores.append(score)
    return scores


def main():
    if OPENAI_API_KEY == "API_KEY":
        raise SystemExit("set OPENAI_API_KEY in .env before running")

    os.environ["WANDB_PROJECT"] = WANDB_PROJECT

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    dataset = dataset.rename_column("completion", "reference")
    dataset = dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_dataset = dataset["train"]

    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_DIR)
    merged_model = AutoModelForCausalLM.from_pretrained(MERGED_MODEL_DIR, dtype="bfloat16")

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type="CAUSAL_LM",
    )
    if INIT_ADAPTER:
        model = PeftModel.from_pretrained(merged_model, INIT_ADAPTER, is_trainable=True)
    else:
        model = get_peft_model(merged_model, lora_config)

    training_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        run_name=RUN_NAME,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_generations=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_LENGTH,
        beta=BETA,
        reward_weights=REWARD_WEIGHTS,
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
        train_dataset=train_dataset,
        processing_class=tokenizer,
        reward_funcs=[task_reward, style_reward, marker_penalty, format_reward],
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "RLAIF_model"))


if __name__ == "__main__":
    main()
