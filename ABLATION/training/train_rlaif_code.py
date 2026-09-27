"""RLAIF GRPO on the ablation mix, starting from the ablation SFT merge rather than the shipped one.
Rewards are untouched: the point is whether coding data in the mix preserves coding, not a new reward."""

import os
import sys

ABLATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ABLATION_DIR)
sys.path.insert(0, REPO_ROOT)

# imported, not copied, so the ablation trains against the same rewards the real stage used
from ABLATION.training.merge_sft_code import SFT_CODE_MERGED_DIR
from RLAIF.training import train_rlaif

DATA_FILE = os.path.join(ABLATION_DIR, "data", "grpo_with_code.jsonl")
OUTPUT_DIR = os.path.join(ABLATION_DIR, "model", "rlaif_code")
RUN_NAME = "qwen2.5-3b-grpo-lora-ablation"


def main():
    if not os.path.exists(DATA_FILE):
        raise SystemExit(f"{DATA_FILE} missing, run ABLATION/data/build_ablation_data.py first")
    if not os.path.isdir(SFT_CODE_MERGED_DIR):
        raise SystemExit(f"{SFT_CODE_MERGED_DIR} missing, run ABLATION/training/merge_sft_code.py first")
    train_rlaif.DATA_FILE = DATA_FILE
    train_rlaif.OUTPUT_DIR = OUTPUT_DIR
    train_rlaif.MERGED_MODEL_DIR = SFT_CODE_MERGED_DIR
    train_rlaif.RUN_NAME = RUN_NAME
    print(f"ablation RLAIF: {DATA_FILE} on {SFT_CODE_MERGED_DIR} -> {OUTPUT_DIR}")
    train_rlaif.main()


if __name__ == "__main__":
    main()
