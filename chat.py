import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = os.path.join(PROJECT_DIR, "outputs", "sft", "final_adapter")

SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def main():
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Type 'quit' to exit, 'reset' to clear conversation history.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("[conversation reset]\n")
            continue

        messages.append({"role": "user", "content": user_input})
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        print(f"Assistant: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
