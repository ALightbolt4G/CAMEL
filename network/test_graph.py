import torch
from graph import ConceptGraph

def test_concept_graph():
    print("--- Testing Concept Graph (Spreading Activation) ---")
    
    domains = ["math", "code", "history", "geography"]
    graph = ConceptGraph(domains=domains, alpha=0.3)
    
    # 1. Test 1-Hop Spread from Math
    print("\n[Test 1] 1-Hop Spread from Math")
    # Math is 1.0, others are 0.0
    initial_A = torch.tensor([1.0, 0.0, 0.0, 0.0])
    
    # Expected: Math connects to Code (weight 0.7).
    # Spread to Code = 1.0 * 0.7 * 0.3 = 0.21
    new_A = graph.spread_activation(initial_A, hops=1)
    
    output_dict = {domains[i]: round(new_A[i].item(), 3) for i in range(len(domains))}
    print(f"Initial: {{'math': 1.0, 'code': 0.0, 'history': 0.0, 'geography': 0.0}}")
    print(f"After Spread: {output_dict}")
    
    assert output_dict["code"] > 0.0, "Activation should spread from math to code"
    assert output_dict["history"] == 0.0, "Activation should NOT spread to unconnected history"
    
    # 2. Test 2-Hop Spread from Math (testing extended reach)
    print("\n[Test 2] 2-Hop Spread from Math")
    # Add a temporary connection code <-> history for this test to show multi-hop spreading
    graph.W[graph.domain_to_idx["code"], graph.domain_to_idx["history"]] = 0.5
    graph.W[graph.domain_to_idx["history"], graph.domain_to_idx["code"]] = 0.5
    
    initial_A2 = torch.tensor([1.0, 0.0, 0.0, 0.0])
    new_A2 = graph.spread_activation(initial_A2, hops=2)
    output_dict2 = {domains[i]: round(new_A2[i].item(), 3) for i in range(len(domains))}
    
    print(f"Initial: {{'math': 1.0, 'code': 0.0, 'history': 0.0, 'geography': 0.0}}")
    print(f"After 2 Hops: {output_dict2}")
    
    assert output_dict2["history"] > 0.0, "Activation should reach history in 2 hops (Math -> Code -> History)"

    print("\n--- All tests completed successfully! ---")

if __name__ == "__main__":
    test_concept_graph()
