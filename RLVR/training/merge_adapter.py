"""Rebuild the RLAIF model as plain weights for RLVR to start from: base + SFT adapter, then + RLAIF adapter.
Writes only under RLVR/model, and also saves the SFT merge so the RLAIF stage can be evaluated on its own base."""

import os
import sys

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RLVR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(RLVR_DIR)
sys.path.insert(0, REPO_ROOT)

from SFT.training.train_sft import MODEL_ID

SFT_ADAPTER = os.path.join(REPO_ROOT, "SFT", "model", "final_adapter")
RLAIF_ADAPTER = os.path.join(REPO_ROOT, "RLAIF", "model", "RLAIF_model")
SFT_MERGED_DIR = os.path.join(RLVR_DIR, "model", "sft_merged")
RLAIF_MERGED_DIR = os.path.join(RLVR_DIR, "model", "rlaif_merged")


def check_adapter(adapter_path):
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    with open(weights_path, "rb") as weights_file:
        if weights_file.read(40).startswith(b"version https://git-lfs"):
            raise SystemExit(f"{weights_path} is a git-lfs pointer, run: git lfs install && git lfs pull")


def main():
    check_adapter(SFT_ADAPTER)
    check_adapter(RLAIF_ADAPTER)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="bfloat16")

    sft_merged = PeftModel.from_pretrained(base_model, SFT_ADAPTER).merge_and_unload()
    sft_merged.save_pretrained(SFT_MERGED_DIR)
    tokenizer.save_pretrained(SFT_MERGED_DIR)
    print(f"wrote {SFT_MERGED_DIR}")

    rlaif_merged = PeftModel.from_pretrained(sft_merged, RLAIF_ADAPTER).merge_and_unload()
    rlaif_merged.save_pretrained(RLAIF_MERGED_DIR)
    tokenizer.save_pretrained(RLAIF_MERGED_DIR)
    print(f"wrote {RLAIF_MERGED_DIR}")


if __name__ == "__main__":
    main()
