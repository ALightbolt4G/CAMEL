import torch
from cell import CamelCell

def test_camel_cell():
    print("--- Testing CAMEL Cell (Tiny Transformer) ---")
    
    # Initialize the cell
    cell = CamelCell(
        vocab_size=32000, 
        d_model=128, 
        n_heads=4, 
        num_layers=4, 
        dim_feedforward=512, 
        max_seq_len=512
    )
    
    # Check parameter count
    param_count = cell.count_parameters()
    print(f"[+] Total Parameters: {param_count:.2f} Million")
    
    # According to MSLM plan: Cell should be 1M - 5M parameters
    assert 1.0 <= param_count <= 6.0, f"Parameter count {param_count}M is out of target range!"
    
    # Create dummy input (batch_size=2, seq_len=10)
    batch_size = 2
    seq_len = 10
    dummy_input = torch.randint(0, 32000, (batch_size, seq_len))
    
    # Create dummy padding mask (last 2 tokens of second sequence are padded)
    padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool)
    padding_mask[1, 8:] = True
    
    # Forward pass
    output = cell(dummy_input, padding_mask=padding_mask)
    
    # Check output shape
    print(f"[+] Output Shape: {output.shape}")
    assert output.shape == (batch_size, 128), "Output shape should be (batch_size, d_model)"
    
    print("--- All tests passed successfully! 🐫 ---")

if __name__ == "__main__":
    test_camel_cell()
