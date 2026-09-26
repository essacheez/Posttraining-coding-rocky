import os
import sys

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

RLAIF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(RLAIF_DIR)
sys.path.insert(0, REPO_ROOT)

from SFT.training.train_sft import MODEL_ID
from SFT.training.train_sft import OUTPUT_DIR as SFT_MODEL_DIR

SFT_ADAPTER = os.path.join(SFT_MODEL_DIR, "final_adapter")
MERGED_MODEL_DIR = os.path.join(RLAIF_DIR, "model", "sft_merged")


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="bfloat16")
    merged = PeftModel.from_pretrained(base_model, SFT_ADAPTER).merge_and_unload()

    merged.save_pretrained(MERGED_MODEL_DIR)
    tokenizer.save_pretrained(MERGED_MODEL_DIR)


if __name__ == "__main__":
    main()
