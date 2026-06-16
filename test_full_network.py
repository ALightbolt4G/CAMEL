import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sys
sys.path.append("/mnt/d/camel")
from router.router import CamelRouter

BASE = "/mnt/d/models/bloom-560m"

print("[*] Loading base model...")
tokenizer = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForCausalLM.from_pretrained(BASE, device_map="auto", torch_dtype=torch.float16)

# Load all adapters
print("[*] Loading all cell adapters...")
model = PeftModel.from_pretrained(base_model, "cells/history_cell/adapter", adapter_name="history")
model.load_adapter("cells/math_cell/adapter", adapter_name="math")
model.load_adapter("cells/code_cell/adapter", adapter_name="code")
model.load_adapter("cells/rezero_cell/adapter", adapter_name="rezero")
model.eval()

# Initialize Router
print("[*] Initializing Biological Router...")
router = CamelRouter()

# Network test queries - mix of simple and complex
test_queries = [
    # Simple domain queries (should work)
    "What is the derivative of x squared?",
    "Who was Napoleon Bonaparte?",
    "Write a Python function to sort a list",
    "What happens to Subaru in Arc 3?",
    # Complex multi-domain queries (historically failed)
    "Write a Python program to calculate the derivative of x squared",
    "Calculate the mathematical probability of WWI starting given the alliance system",
    "Explain how recursion in programming is similar to mathematical induction",
    "Write a Python simulation of World War II battle outcomes",
]

adapter_map = {
    "math_cell": "math",
    "code_cell": "code",
    "history_cell": "history",
    "rezero_cell": "rezero",
}

print()
print("=" * 70)
print(" FULL NETWORK TEST (Router + Cell Activation)")
print("=" * 70)

for q in test_queries:
    active_cells, scores = router.route(q)
    
    print(f"\nQuery: \"{q}\"")
    print(f"Router Scores: {{{', '.join(f'{k}: {v:.4f}' for k, v in sorted(scores.items(), key=lambda x: -x[1]))}}}")
    print(f"Active Cells (>0.2): {active_cells}")
    
    if active_cells:
        # Use the highest-scoring cell
        best_cell = active_cells[0]
        adapter_name = adapter_map.get(best_cell, None)
        if adapter_name:
            model.set_adapter(adapter_name)
            inputs = tokenizer(q, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    temperature=0.5,
                    top_p=0.9,
                    do_sample=True,
                    repetition_penalty=1.2,
                    pad_token_id=tokenizer.eos_token_id
                )
            response = tokenizer.decode(out[0], skip_special_tokens=True)
            answer = response[len(q):].strip()
            print(f"Activated: {best_cell} | Output: {answer[:250]}")
    else:
        print("ROUTING FAILED: No cell crossed 0.2 threshold. Falling back to base model.")
        inputs = tokenizer(q, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = base_model.generate(
                **inputs,
                max_new_tokens=80,
                temperature=0.5,
                pad_token_id=tokenizer.eos_token_id
            )
        response = tokenizer.decode(out[0], skip_special_tokens=True)
        answer = response[len(q):].strip()
        print(f"Base Model Fallback: {answer[:250]}")
    
    print("-" * 70)
