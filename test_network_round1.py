import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings
from router.router import CamelRouter

warnings.filterwarnings('ignore')

print("Initializing MSLM Network Evaluation - REAL INFERENCE ROUND 1...")
base_model_name = "/mnt/d/models/bloom-560m"
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
except Exception:
    print("Falling back to huggingface hub...")
    base_model_name = "bigscience/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

print("Loading Adapters into Network...")
model = PeftModel.from_pretrained(base_model, "cells/code_cell/adapter", adapter_name="code_cell")
model.load_adapter("cells/math_cell/adapter", adapter_name="math_cell")
model.load_adapter("cells/history_cell/adapter", adapter_name="history_cell")

router = CamelRouter()

queries = [
    "Write a Python program to calculate the derivative of x squared",
    "Write a Python simulation of World War II battle outcomes",
    "Calculate the mathematical probability of WWI starting given the alliance system"
]

print("\n=== NETWORK TESTS - Round 1 (REAL INFERENCE) ===\n")
for i, query in enumerate(queries):
    print(f"\n[{i+1}/3] Query: {query}")
    active_cells, scores = router.route(query)
    print(f"--> Active Cells: {active_cells}")
    
    if len(active_cells) == 0:
        continue
        
    try:
        if len(active_cells) > 1:
            weights = [1.0] * len(active_cells)
            model.add_weighted_adapter(active_cells, weights, "current_merged", combination_type="linear")
            model.set_adapter("current_merged")
        else:
            model.set_adapter(active_cells[0])
            
        inputs = tokenizer(query, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"--> OUTPUT:\n{response.replace(query, '').strip()}\n")
        
        if len(active_cells) > 1:
            model.delete_adapter("current_merged")
            
    except Exception as e:
        print(f"--> Generation failed: {e}")
