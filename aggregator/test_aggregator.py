import torch
import torch.nn as nn
from aggregator import CamelAggregator

def test_aggregator():
    print("--- Testing Camel Aggregator (Weight Tying) ---")
    
    batch_size = 2
    seq_len = 5
    embed_dim = 128
    vocab_size = 32000
    
    # 1. Create a dummy shared embedding weight
    shared_embedding = nn.Embedding(vocab_size, embed_dim)
    
    # 2. Initialize Aggregator
    aggregator = CamelAggregator(shared_embedding.weight)
    
    print(f"Aggregator initialized. LM Head weights shape: {aggregator.lm_head.weight.shape}")
    
    # Check weight tying mechanism
    assert torch.equal(aggregator.lm_head.weight, shared_embedding.weight), "Weight tying failed! Memory is being wasted."
    print("[+] Weight Tying is working correctly (0 extra parameters).")
    
    # 3. Mock active cell outputs (Tensors representing sequence embeddings)
    cell_outputs = {
        "math": torch.randn(batch_size, seq_len, embed_dim),
        "code": torch.randn(batch_size, seq_len, embed_dim),
        "history": torch.randn(batch_size, seq_len, embed_dim)
    }
    
    # 4. Mock final activation scores (From the Router & Graph)
    activation_scores = {
        "math": 0.8,
        "code": 0.5,
        "history": 0.0 # Below threshold, should have zero impact
    }
    
    # 5. Forward pass
    logits = aggregator(cell_outputs, activation_scores)
    
    print(f"Cell outputs shape: {cell_outputs['math'].shape}")
    print(f"Final Logits shape: {logits.shape}")
    
    assert logits.shape == (batch_size, seq_len, vocab_size), "Logits shape is incorrect."
    
    # Verify math and code contributed properly, and history was ignored
    manual_agg = (cell_outputs["math"] * 0.8) + (cell_outputs["code"] * 0.5)
    manual_logits = torch.matmul(manual_agg, shared_embedding.weight.T)
    
    assert torch.allclose(logits, manual_logits, atol=1e-5), "Aggregation mathematical logic is incorrect."
    
    print("\n--- All tests completed successfully! 🧬 ---")

if __name__ == "__main__":
    test_aggregator()
