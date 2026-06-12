import json
import os
import random

def generate_synthetic_dataset(output_path="data/synthetic_dataset.json"):
    """
    Generates a 2200-sentence synthetic dataset across 4 pure domains and 1 hybrid domain.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Pure Domains
    math_templates = [
        "calculate the integral of {x} dx", "what is {n} + {m} equals",
        "solve the matrix equation for {var}", "the derivative of {var} is"
    ]
    code_templates = [
        "def {func}(): return {val}", "import {lib} as {alias}",
        "struct {name} {{ int x; }};", "class {cls}(object): pass"
    ]
    history_templates = [
        "the war of {year} was fought by {king}", "in the {century} century, the empire fell",
        "the ancient battle took place near {place}", "emperor {king} conquered the lands"
    ]
    geography_templates = [
        "the river {river} flows through {country}", "the capital of {country} is {city}",
        "the {mountain} mountains are located in {continent}", "look at the map of {region}"
    ]
    
    # Hybrid Domain (Code + Math, History + Geography)
    hybrid_templates = [
        "write a python script to calculate the matrix determinant",
        "def calculate_area(radius): return 3.14 * radius * radius",
        "the map of the roman empire in the 1st century",
        "the historical borders of the nile river country"
    ]
    
    dataset = []
    
    # Generate 500 pure domain sentences by repeating and randomizing templates
    for _ in range(125):
        for t in math_templates: dataset.append({"text": t.format(x="x^2", n=5, m=3, var="y"), "domain": "math"})
        for t in code_templates: dataset.append({"text": t.format(func="main", val="0", lib="sys", alias="s", name="Node", cls="Model"), "domain": "code"})
        for t in history_templates: dataset.append({"text": t.format(year="1945", king="Arthur", century="19th", place="Waterloo"), "domain": "history"})
        for t in geography_templates: dataset.append({"text": t.format(river="Nile", country="Egypt", city="Cairo", mountain="Alps", continent="Europe", region="Asia"), "domain": "geography"})
        
    # Generate 200 hybrid sentences
    for _ in range(50):
        for t in hybrid_templates: dataset.append({"text": t, "domain": "hybrid"})
        
    # Shuffle dataset
    random.shuffle(dataset)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
        
    print(f"Generated {len(dataset)} synthetic sentences at {output_path}")

if __name__ == "__main__":
    generate_synthetic_dataset()
