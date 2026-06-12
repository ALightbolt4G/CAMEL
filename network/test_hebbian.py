import torch
from graph import ConceptGraph
from hebbian import HebbianUpdater

def test_hebbian_learning():
    print("--- Testing Hebbian Learning (v0.6.0) ---")
    
    domains = ["math", "code", "history", "geography"]
    graph = ConceptGraph(domains=domains, alpha=0.2)
    updater = HebbianUpdater(learning_rate=0.1, decay_rate=0.02)
    
    # Example: User asks "Astronomical calculation" -> Triggers Math and Geography.
    idx_m = graph.domain_to_idx["math"]
    idx_g = graph.domain_to_idx["geography"]
    
    print(f"Initial Weight (Math <-> Geography): {graph.W[idx_m, idx_g]:.3f}")
    
    # Simulate co-activation: Math=0.9, Geography=0.8, others=0.0
    activations = torch.tensor([0.9, 0.0, 0.0, 0.8])
    
    # 1. First Update
    graph.W = updater.update_weights(graph.W, activations)
    print(f"Weight after 1st Co-activation: {graph.W[idx_m, idx_g]:.3f}")
    
    # 2. Second Update
    graph.W = updater.update_weights(graph.W, activations)
    print(f"Weight after 2nd Co-activation: {graph.W[idx_m, idx_g]:.3f}")
    
    assert graph.W[idx_m, idx_g] > 0.0, "Weight should increase due to Hebbian learning"
    
    # 3. Simulate decay (no co-activation)
    zero_activations = torch.tensor([0.0, 1.0, 0.0, 0.0]) # Only code fires
    graph.W = updater.update_weights(graph.W, zero_activations)
    print(f"Weight after Decay (No Math/Geo activation): {graph.W[idx_m, idx_g]:.3f}")
    
    # Ensure self-connections are 0
    assert graph.W[idx_m, idx_m] == 0.0, "Self connection must remain 0"
    
    print("\n--- All tests completed successfully! ---")

if __name__ == "__main__":
    test_hebbian_learning()
