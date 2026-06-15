import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/mnt/d/models/bloom-560m"
HISTORY_ADAPTER = "cells/history_cell/adapter"
REZERO_ADAPTER = "cells/rezero_cell/adapter"

print("[*] Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("[*] Loading Adapters...")
model = PeftModel.from_pretrained(base_model, HISTORY_ADAPTER, adapter_name="history")
model.load_adapter(REZERO_ADAPTER, adapter_name="rezero")

print("[*] Creating Fusion Adapter (50% History + 50% ReZero)...")
try:
    model.add_weighted_adapter(
        adapters=["history", "rezero"],
        weights=[0.5, 0.5],
        adapter_name="fusion",
        combination_type="linear"
    )
except Exception as e:
    print(f"[!] Error creating fusion adapter: {e}")
    exit(1)

prompts = [
    "The great war that destroyed the capital city was started by",
    "Natsuki Subaru walked into the battlefield and saw"
]

for prompt in prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print("\n" + "="*70)
    print(f"[Prompt]: {prompt}")
    print("="*70)
    
    with torch.no_grad():
        for adapter in ["history", "rezero", "fusion"]:
            model.set_adapter(adapter)
            out = model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id
            )
            gen = tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"[{adapter.upper()} CELL]:")
            print(gen[len(prompt):].strip())
            print("-" * 70)
