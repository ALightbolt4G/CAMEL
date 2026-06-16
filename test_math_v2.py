import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "/mnt/d/models/bloom-560m"
ADAPTER = "cells/math_cell/adapter"

print("[*] Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype=torch.float16)

print("[*] Loading math_cell adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

questions = [
    "What is the derivative of x^3?",
    "Solve for x: 5x + 3 = 18",
    "What is the integral of 2x dx?",
    "Explain the Pythagorean theorem",
    "What is a prime number?",
    "What is the fundamental theorem of calculus?",
    "Solve the quadratic equation x^2 - 5x + 6 = 0",
]

print()
print("="*60)
print(" MATH_CELL v2 - Quality Test")
print("="*60)

for q in questions:
    inputs = tokenizer(q, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.5,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(out[0], skip_special_tokens=True)
    answer = response[len(q):].strip()
    print(f"\nQ: {q}")
    print(f"A: {answer[:400]}")
    print("-"*60)
