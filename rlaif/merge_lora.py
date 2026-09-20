"""Merges the SFT LoRA adapter into the base weights, and checks the merge held.

slime trains full-parameter through Megatron and has no LoRA support, so GRPO
starts from merged weights. This is a format change, not a training step: the
merged model has to be the same model checkpoint 1 measured.

    uv run python rlaif/merge_lora.py
    uv run python rlaif/merge_lora.py --verify
"""

import argparse
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# run as a script, sys.path[0] is rlaif/, which leaves the repo-root modules out
sys.path.insert(0, PROJECT_DIR)

ADAPTER_DIR = os.path.join(PROJECT_DIR, "outputs", "sft", "final_adapter")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "sft", "merged_hf")

# what checkpoint 1 measured for this adapter served as base + LoRA, from
# EVAL_RESULTS.md. --verify compares against these; style_cosine is computed over
# a whole set of responses, so it only means anything at the same prompt count.
SFT_STYLE_COSINE = 0.862
SFT_MEAN_WORDS = 101
SFT_EVAL_PROMPTS = 200


def merge(adapter_path, output_path):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from evaluate_persona import check_adapter
    from train_sft import MODEL_ID

    # the adapter ships via git-lfs, so a plain clone leaves a pointer file that
    # would merge into a silent no-op and cost a whole RL run to notice
    check_adapter(adapter_path)

    print(f"loading {MODEL_ID}")
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="bfloat16")
    print(f"applying {adapter_path}")
    merged = PeftModel.from_pretrained(base, adapter_path).merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path)
    # slime and both eval harnesses read the tokenizer from the model directory
    AutoTokenizer.from_pretrained(MODEL_ID).save_pretrained(output_path)
    print(f"wrote merged model to {output_path}")


def verify(output_path, num_prompts, group_size):
    """Two pre-flight checks on one model load, both gating the GRPO run.

    Did the merge hold the persona? A bad merge gives a fluent model that has
    quietly lost the voice, which every later stage would read as an RLAIF
    result rather than a broken starting point.

    Does GRPO have a gradient? The advantage is taken within a group, so the
    spread that matters is within-group, not the across-prompt spread in
    EVAL_RESULTS.md section 4. Flat groups mean every advantage is ~0.
    """
    import statistics

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from evaluate_persona import (
        MAX_NEW_TOKENS,
        build_eval_prompt,
        cosine,
        load_eval_examples,
        load_persona_corpus,
        profile_vector,
    )
    from judge import build_completer, reward, score_responses
    from rlaif.reward import JUDGE_MODEL

    tokenizer = AutoTokenizer.from_pretrained(output_path)
    llm = LLM(model=output_path, dtype="bfloat16")
    examples = load_eval_examples(num_prompts)
    prompts = [build_eval_prompt(tokenizer, example["prompt"]) for example in examples]

    greedy = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
    texts = [output.outputs[0].text.strip() for output in greedy]
    style_cosine = cosine(profile_vector(texts), profile_vector(load_persona_corpus()))
    mean_words = statistics.mean(len(text.split()) for text in texts)

    print(f"\n=== merge check ({len(texts)} prompts, greedy) ===")
    print(f"style_cosine        {style_cosine:.3f}")
    print(f"mean response words {mean_words:.0f}")
    if len(texts) == SFT_EVAL_PROMPTS:
        print(f"checkpoint 1, same {SFT_EVAL_PROMPTS} prompts as base + LoRA: "
              f"style_cosine {SFT_STYLE_COSINE}, mean words {SFT_MEAN_WORDS}")
    else:
        print(f"note: checkpoint 1's {SFT_STYLE_COSINE} was over {SFT_EVAL_PROMPTS} prompts. "
              f"style_cosine is a whole-set profile and rises with sample size, so it is "
              f"only comparable at --num-prompts {SFT_EVAL_PROMPTS}.")
    print("\nsample:\n" + texts[0][:300])

    probe = prompts[: max(1, num_prompts // 4)]
    sampled = llm.generate(
        probe, SamplingParams(temperature=1.0, max_tokens=MAX_NEW_TOKENS, n=group_size)
    )
    complete = build_completer(JUDGE_MODEL, max_retries=8)
    stds, means = [], []
    for output in sampled:
        group = [completion.text.strip() for completion in output.outputs]
        scored = score_responses(group, complete, max_workers=group_size, quiet=True)
        totals = [reward(scores) for scores in scored if scores]
        if len(totals) > 1:
            stds.append(statistics.pstdev(totals))
            means.append(statistics.mean(totals))

    print(f"\n=== GRPO signal ({len(stds)} groups of {group_size}, temperature 1.0) ===")
    print(f"within-group std   mean {statistics.mean(stds):.2f}  median {statistics.median(stds):.2f}  "
          f"min {min(stds):.2f}  max {max(stds):.2f}")
    print(f"flat groups        {sum(std == 0 for std in stds)}/{len(stds)}  (no gradient)")
    print(f"group mean total   {statistics.mean(means):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Merge the SFT LoRA adapter for RLAIF")
    parser.add_argument("--adapter", default=ADAPTER_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="check the merged model against checkpoint 1 and probe the GRPO signal",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=SFT_EVAL_PROMPTS,
        help=f"prompts for --verify, default {SFT_EVAL_PROMPTS} to match checkpoint 1",
    )
    parser.add_argument(
        "--group-size", type=int, default=8, help="samples per prompt for the signal probe"
    )
    args = parser.parse_args()

    if args.verify:
        verify(args.output, args.num_prompts, args.group_size)
    else:
        merge(args.adapter, args.output)


if __name__ == "__main__":
    main()
