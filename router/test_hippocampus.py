from hippocampus import Hippocampus

def test_hippocampus():
    print("--- Testing Hippocampus (Level 3 Router) ---")
    
    hippo = Hippocampus(decay_factor=0.7, boost_weight=0.3, switch_threshold=0.7)
    
    # 1. Build context gradually in History
    print("\n[Test 1] Building Context in History")
    history_scores = [
        {"history": 0.4, "math": 0.0, "code": 0.0},
        {"history": 0.5, "math": 0.0, "code": 0.0},
        {"history": 0.6, "math": 0.0, "code": 0.0}
    ]
    
    for i, scores in enumerate(history_scores):
        final = hippo.apply_context(scores)
        print(f"Msg {i+1} Output:", {k: round(v, 3) for k, v in final.items()})
        
    print("Memory State after Msg 3:", {k: round(v, 3) for k, v in hippo.memory_state.items()})
    
    # 2. Ambiguous query (benefits from History boost)
    print("\n[Test 2] Ambiguous query benefiting from context")
    ambiguous_scores = {"history": 0.3, "math": 0.3, "code": 0.0}
    final_ambiguous = hippo.apply_context(ambiguous_scores)
    print("Input:", ambiguous_scores)
    print("Output:", {k: round(v, 3) for k, v in final_ambiguous.items()})
    assert final_ambiguous["history"] > final_ambiguous["math"], "History should get a boost from memory"
    
    # 3. Sudden Context Switch to Math
    print("\n[Test 3] Sudden Context Switch (Attention Reset)")
    switch_scores = {"history": 0.0, "math": 0.85, "code": 0.0}
    final_switch = hippo.apply_context(switch_scores)
    print("Input:", switch_scores)
    print("Output:", {k: round(v, 3) for k, v in final_switch.items()})
    print("Memory State after Switch:", {k: round(v, 3) for k, v in hippo.memory_state.items()})
    
    # Ensure memory of history was flushed
    assert hippo.memory_state["history"] < 0.1, "History memory should be flushed after a strong context switch"

    print("\n--- All tests completed successfully! ---")

if __name__ == "__main__":
    test_hippocampus()
