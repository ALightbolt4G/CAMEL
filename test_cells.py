import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "/mnt/d/models/bloom-560m"
try:
    print(f"Loading Base Model: {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
except Exception as e:
    print(f"Failed to load model from {base_model_name}, trying huggingface hub...")
    base_model_name = "bigscience/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

questions = {
    "code_cell": [
        "What is a Python list comprehension?",
        "Explain how Python functions work",
        "What is the difference between a list and a tuple?"
    ],
    "history_cell": [
        "What caused World War I?",
        "Who was Adolf Hitler?",
        "What happened at the end of World War II?"
    ],
    "math_cell": [
        "What is a derivative in calculus?",
        "Explain linear algebra basics",
        "What is the Pythagorean theorem?"
    ]
}

for cell_name, cell_questions in questions.items():
    print(f"\n{'='*50}\nTesting {cell_name.upper()}\n{'='*50}")
    adapter_path = f"cells/{cell_name}/adapter"
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
        print(f"[Loaded Adapter: {adapter_path}]")
    except Exception as e:
        print(f"Could not load adapter {adapter_path}: {e}")
        continue
        
    for q in cell_questions:
        print(f"\nQ: {q}")
        inputs = tokenizer(q, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"A: {response.replace(q, '').strip()}")
