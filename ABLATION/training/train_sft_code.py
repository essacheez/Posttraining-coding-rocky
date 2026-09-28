"""SFT on the ablation mix, which swaps a quarter of the persona rows for MBPP coding rows.
Patches only the data and output paths, so the recipe stays identical to the shipped SFT stage."""

import os
import sys

ABLATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ABLATION_DIR)
sys.path.insert(0, REPO_ROOT)

# imported, not copied, so the ablation trains with the same code the real stage used
from SFT.training import train_sft

# VARIANT picks which coding targets to train on: mbpp reference solutions or the model's own
VARIANT = os.environ.get("VARIANT", "code")
DATA_FILE = os.path.join(ABLATION_DIR, "data", f"sft_with_{'selfgen' if VARIANT == 'selfgen' else 'code'}.jsonl")
OUTPUT_DIR = os.path.join(ABLATION_DIR, "model", f"sft_{VARIANT}")
RUN_NAME = f"qwen2.5-3b-sft-lora-ablation-{VARIANT}"


def main():
    if not os.path.exists(DATA_FILE):
        raise SystemExit(f"{DATA_FILE} missing, run ABLATION/data/build_ablation_data.py first")
    train_sft.DATA_FILE = DATA_FILE
    train_sft.OUTPUT_DIR = OUTPUT_DIR
    train_sft.RUN_NAME = RUN_NAME
    print(f"ablation SFT: {DATA_FILE} -> {OUTPUT_DIR}")
    train_sft.main()


if __name__ == "__main__":
    main()
