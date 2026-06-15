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

prompts = [
    # 1. سؤال (QA)
    "Q: Who is Rem, and what is her relationship with Subaru? A:",
    
    # 2. تكملة نص (Text Completion)
    "Subaru walked into the Sanctuary and saw Echidna sipping tea. She smiled and said,",
    
    # 3. توقع للمستقبل (Arc 6 Prediction)
    "In Arc 6, at the Pleiades Watchtower, Subaru realizes that the only way to save everyone is to"
]

print("\n" + "="*50)
print(" Re:Zero Cell - Advanced Testing")
print("="*50 + "\n")

with torch.no_grad():
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Test {i+1}:")
        print(f"[Prompt]: {prompt}")
        print(f"[Generated]: {generated_text[len(prompt):].strip()}")
        print("-" * 50)
