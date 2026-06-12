import torch
import torch.nn as nn
from prefrontal import PrefrontalCortex

def test_prefrontal_cortex():
    print("--- Testing Prefrontal Cortex (Level 2 Router) ---")
    
    # 1. Setup Mock Shared Embedding Layer
    vocab_size = 1000
    embed_dim = 128
    shared_embedding = nn.Embedding(vocab_size, embed_dim)
    
    # 2. Setup Mock Domain Centroids
    domains = ["math", "code", "history", "geography"]
    domain_centroids = {}
    for d in domains:
        # Create a random normalized vector for each domain
        vec = torch.randn(embed_dim)
        domain_centroids[d] = vec / vec.norm()
        
    # Initialize PFC
    pfc = PrefrontalCortex(shared_embedding, domain_centroids)
    
    # --- Test Case 1: High Thalamus Confidence ---
    print("\n[Test 1] High Thalamus Confidence (Bypass PFC)")
    query_tokens = torch.randint(0, vocab_size, (1, 10))
    thalamus_hint_confident = {"math": 0.9, "code": 0.1, "history": 0.0, "geography": 0.0}
    
    scores1 = pfc.evaluate(query_tokens, thalamus_hint_confident)
    print(f"Input Hint: {thalamus_hint_confident}")
    print(f"Output Scores: {scores1}")
    assert scores1 == thalamus_hint_confident, "PFC should be bypassed when Thalamus is highly confident"
    
    # --- Test Case 2: Low Thalamus Confidence (Engage PFC) ---
    print("\n[Test 2] Low Thalamus Confidence (Engage PFC)")
    query_tokens = torch.randint(0, vocab_size, (1, 10))
    thalamus_hint_uncertain = {"math": 0.33, "code": 0.33, "history": 0.0, "geography": 0.0}
    
    scores2 = pfc.evaluate(query_tokens, thalamus_hint_uncertain)
    print(f"Input Hint: {thalamus_hint_uncertain}")
    print("Output Scores:", {k: round(v, 3) for k, v in scores2.items()})
    
    print("\n--- All tests completed successfully! 🧠 ---")

if __name__ == "__main__":
    test_prefrontal_cortex()
