import sys
import os

# Ensure the router module can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from router.router import CamelRouter

def evaluate_router():
    print("Initializing CamelRouter (Loading all-MiniLM-L6-v2)...")
    router = CamelRouter()
    
    queries = [
        # Round 1
        "Write a Python program to calculate the derivative of x squared",
        "Write a Python simulation of World War II battle outcomes",
        "Calculate the mathematical probability of WWI starting given the alliance system",
        
        # Round 2
        "Write a Python function to model population growth using differential equations",
        "Calculate the statistical probability of WWI given European alliance mathematics",
        "Write a Python program that simulates the economic impact of WWII using graphs",
        "Explain how recursion in programming is similar to mathematical induction"
    ]
    
    # Expected primary cell for each query
    expected = [
        "code_cell",     # 1: Python program
        "code_cell",     # 2: Python simulation
        "math_cell",     # 3: mathematical probability
        "code_cell",     # 4: Python function
        "math_cell",     # 5: statistical probability
        "code_cell",     # 6: Python program
        "code_cell"      # 7: programming recursion
    ]
    
    # Secondary/multi-cell expected (optional but good to see)
    expected_secondary = [
        "math_cell",     # derivative of x squared
        "history_cell",  # World War II
        "history_cell",  # WWI alliance
        "math_cell",     # differential equations
        "history_cell",  # WWI European alliance
        "history_cell",  # WWII economic impact
        "math_cell"      # mathematical induction
    ]
    
    print("\n" + "="*50)
    print("ROUTER EVALUATION ON 7 QUERIES")
    print("="*50)
    
    success_count = 0
    total = len(queries)
    
    for i, query in enumerate(queries):
        active_cells, scores = router.route(query)
        
        # Determine the top activated cell
        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_cell = sorted_scores[0][0]
        
        expected_top = expected[i]
        expected_sec = expected_secondary[i]
        
        is_success = top_cell == expected_top
        if is_success:
            success_count += 1
            
        print(f"\nQuery {i+1}: '{query}'")
        print(f"Top Score Cell: {top_cell} | Expected Top: {expected_top}")
        print(f"Active Cells (>0.2): {active_cells}")
        print(f"All Scores: { {k: round(v, 4) for k, v in scores.items()} }")
        print(f"Success (Primary Target): {'✅' if is_success else '❌'}")
        
    print("\n" + "="*50)
    success_rate = (success_count / total) * 100
    print(f"ROUTER ACCURACY: {success_count}/{total} ({success_rate:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    evaluate_router()
