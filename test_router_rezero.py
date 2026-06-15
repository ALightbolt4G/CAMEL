import sys
sys.path.append("/mnt/d/camel")

from router.router import CamelRouter

def main():
    print("[*] Initializing Biological Router (Thalamus + Prefrontal Cortex)...")
    router = CamelRouter()
    
    test_queries = [
        "In Arc 6, what happens to Subaru at the Pleiades Watchtower?",
        "Explain the history of the Witch of Envy.",
        "Write a python script to simulate Return by Death loops.",
        "The great war that destroyed the capital city was started by",
        "Who is Rem and why did she attack Subaru in Arc 2?"
    ]
    
    print("\n" + "="*50)
    print(" Router Evaluation: Re:Zero Cell Integration")
    print("="*50)
    
    for q in test_queries:
        print(f"\n[Query]: {q}")
        
        # We need to look inside the router's decision process
        thalamus_scores = router.thalamus.route(q)
        pfc_scores = router.prefrontal.evaluate(q)
        
        # Combine them as the router does
        final_scores = {}
        # Ensure all domains from PFC are covered
        for domain in pfc_scores.keys():
            t_score = thalamus_scores.get(domain, 0.0)
            p_score = pfc_scores.get(domain, 0.0)
            final_scores[domain] = (t_score * 0.3) + (p_score * 0.7)
            
        print("[Thalamus Scores]:", {k: round(v, 4) for k, v in thalamus_scores.items() if v > 0})
        print("[PFC Scores]:", {k: round(v, 4) for k, v in pfc_scores.items() if v > 0.1})
        print("[Final Scores]:", {k: round(v, 4) for k, v in final_scores.items() if v > 0.1})
        
        best_domain = max(final_scores.items(), key=lambda x: x[1])
        print(f"[Selected Cell]: {best_domain[0]}_cell (Score: {best_domain[1]:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    main()
