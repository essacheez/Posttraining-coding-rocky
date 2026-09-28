"""Merge the ablation SFT adapter into plain weights, the base the ablation RLAIF stage trains on.
Mirrors RLVR/training/merge_adapter.py but writes only under ABLATION/model."""

import os
import sys

ABLATION_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(ABLATION_DIR)
sys.path.insert(0, REPO_ROOT)

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from SFT.training.train_sft import MODEL_ID

SFT_CODE_ADAPTER = os.path.join(ABLATION_DIR, "model", "sft_code", "final_adapter")
SFT_CODE_MERGED_DIR = os.path.join(ABLATION_DIR, "model", "sft_code_merged")


def main():
    if not os.path.isdir(SFT_CODE_ADAPTER):
        raise SystemExit(f"{SFT_CODE_ADAPTER} missing, run ABLATION/training/train_sft_code.py first")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="bfloat16")
    merged = PeftModel.from_pretrained(base_model, SFT_CODE_ADAPTER).merge_and_unload()
    merged.save_pretrained(SFT_CODE_MERGED_DIR)
    tokenizer.save_pretrained(SFT_CODE_MERGED_DIR)
    print(f"wrote {SFT_CODE_MERGED_DIR}")


if __name__ == "__main__":
    main()
