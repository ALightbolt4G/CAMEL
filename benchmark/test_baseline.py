import torch
from baseline import BaselineTransformer

def test_baseline_size():
    print("--- Testing Baseline Model (Monolithic Dense Transformer) ---")
    
    # Initialize Baseline
    # Trying to hit ~20M params: vocab=32k, embed_dim=256, layers=8, heads=8
    model = BaselineTransformer(
        vocab_size=32000, 
        d_model=256, 
        n_heads=8, 
        num_layers=8, 
        dim_feedforward=1024
    )
    
    param_count = model.count_parameters()
    print(f"[+] Baseline Model Parameters: {param_count:.2f} Million")
    
    # CAMEL Cell is ~4.9M. 4 Cells = 19.6M. 
    # Baseline should be around 19 - 22M.
    assert 18.0 <= param_count <= 25.0, f"Baseline parameter count {param_count}M is not within the expected Apple-to-Apple comparison range!"
    
    print("\n--- Model Size Check Passed Successfully! ⚖️ ---")

if __name__ == "__main__":
    test_baseline_size()
