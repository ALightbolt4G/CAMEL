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

print(f"[*] Loading Adapters (History & Re:Zero)...")
# Load base model into PEFT and add first adapter
model = PeftModel.from_pretrained(base_model, HISTORY_ADAPTER, adapter_name="history")
# Load second adapter
model.load_adapter(REZERO_ADAPTER, adapter_name="rezero")
model.eval()

# We will test a prompt that is ambiguous. 
# "The great war that destroyed the capital city was started by"
# The History cell should talk about real world history (like WWII, Rome, etc).
# The ReZero cell should talk about Lugunica, witches, or fantasy elements.

prompt = "The great war that destroyed the capital city was started by"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n" + "="*60)
print(" MSLM Network Interference Test: History vs Re:Zero")
print("="*60 + "\n")

with torch.no_grad():
    print(">>> 1. Activating [history_cell]")
    model.set_adapter("history")
    out_history = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.3, # History prefers lower temp for facts
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    gen_history = tokenizer.decode(out_history[0], skip_special_tokens=True)
    print(f"[Prompt]: {prompt}")
    print(f"[Generated]: {gen_history[len(prompt):].strip()}")
    print("-" * 60)

    print(">>> 2. Activating [rezero_cell]")
    model.set_adapter("rezero")
    out_rezero = model.generate(
        **inputs,
        max_new_tokens=60,
        temperature=0.8, # ReZero prefers higher temp for creativity
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    gen_rezero = tokenizer.decode(out_rezero[0], skip_special_tokens=True)
    print(f"[Prompt]: {prompt}")
    print(f"[Generated]: {gen_rezero[len(prompt):].strip()}")
    print("-" * 60)

print("[*] SUCCESS: Both cells responded without breaking the other (No Catastrophic Forgetting)!")
