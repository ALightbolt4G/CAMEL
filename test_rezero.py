import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/mnt/d/models/bloom-560m"
ADAPTER_PATH = "cells/rezero_cell/adapter"

print("[*] Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)

print(f"[*] Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

prompt = "Subaru looked at Emilia with a serious expression and said,"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("\n[*] Generating text (Re:Zero Cell)...\n")
print(f"Prompt: {prompt}\n")
print("-" * 50)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
print("-" * 50)
