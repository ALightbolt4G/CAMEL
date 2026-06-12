from thalamus import Thalamus

def test_thalamus():
    print("--- Testing Thalamus (Level 1 Router) ---")
    
    thalamus = Thalamus()
    
    queries = [
        "أريد كتابة كود Python يستخدم def و return",
        "متى وقعت الحرب العالمية الثانية ومن هو الملك؟",
        "calculate the matrix equation 5 + 3 equals what?",
        "أين يقع نهر النيل في خريطة العالم؟",
        "كيف أتعلم استخدام matrix لعمل كود؟", # Mixed domains
        "لا أعرف ماذا أقول." # Unrelated
    ]
    
    for q in queries:
        print(f"\nQuery: '{q}'")
        scores = thalamus.route(q)
        # Sort and display
        sorted_scores = {k: f"{v:.2f}" for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        print(f"Domain Hint: {sorted_scores}")
        
    print("\n--- All tests completed successfully! 🧠 ---")

if __name__ == "__main__":
    test_thalamus()
