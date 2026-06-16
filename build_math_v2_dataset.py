import json
import random
import os
import requests
import fitz  # PyMuPDF
import sympy as sp
from tqdm import tqdm

DATA_DIR = "data/math_cell"
os.makedirs(DATA_DIR, exist_ok=True)
OUT_FILE = os.path.join(DATA_DIR, "math_v2_data.json")

PDF_SOURCES = {
    "Polya_HowToSolveIt": "https://www.hlevkin.com/hlevkin/90MathPhysBioBooks/Math/Polya/George_Polya_How_To_Solve_It_.pdf",
    "Zeitz_ArtOfProblemSolving": "https://kheavan.wordpress.com/wp-content/uploads/2010/06/paul-zeitz-author-the-art-and-craft-of-problem-solving-2edwiley20060471789011.pdf",
    "Thomas_Calculus": "https://rodrigopacios.github.io/mrpacios/download/Thomas_Calculus.pdf"
}

def download_pdf(url, name):
    path = os.path.join(DATA_DIR, f"{name}.pdf")
    if os.path.exists(path):
        print(f"[*] {name} already downloaded.")
        return path
    print(f"[*] Downloading {name} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return path
    except Exception as e:
        print(f"[!] Failed to download {name}: {e}")
        return None

def extract_text_from_pdf(pdf_path, max_pages=150):
    """Extract descriptive text from PDF books."""
    text_chunks = []
    if not pdf_path or not os.path.exists(pdf_path):
        return text_chunks
    print(f"[*] Extracting text from {pdf_path}...")
    try:
        doc = fitz.open(pdf_path)
        for i in range(min(max_pages, len(doc))):
            page = doc.load_page(i)
            text = page.get_text("text").strip()
            if len(text) > 200:
                # Split by double newline to get rough paragraphs
                paragraphs = text.split("\n\n")
                for p in paragraphs:
                    clean_p = p.replace("\n", " ").strip()
                    if len(clean_p) > 100:
                        text_chunks.append({"text": clean_p, "source": os.path.basename(pdf_path)})
    except Exception as e:
        print(f"[!] Error parsing PDF: {e}")
    return text_chunks

def generate_algebra_problems(count=1500):
    """Generate linear and quadratic algebra problems."""
    chunks = []
    x = sp.Symbol('x')
    print("[*] Generating Algebra Problems...")
    for _ in range(count):
        a = random.randint(1, 20)
        b = random.randint(-20, 20)
        c = random.randint(-20, 20)
        
        # Linear: ax + b = c
        eq = sp.Eq(a*x + b, c)
        solution = sp.solve(eq, x)
        if solution:
            ans = solution[0]
            q = f"Q: Solve for x in the equation {a}x + ({b}) = {c}\n"
            a_text = f"A: First, subtract {b} from both sides to get {a}x = {c - b}. Then, divide by {a}. The solution is x = {ans}."
            chunks.append({"text": q + a_text, "source": "synthetic_algebra"})
            
        # Quadratic: x^2 + bx + c = 0
        eq2 = sp.Eq(x**2 + b*x + c, 0)
        sol2 = sp.solve(eq2, x)
        if sol2:
            q2 = f"Q: Find the roots of the quadratic equation x^2 + ({b})x + ({c}) = 0\n"
            a2 = f"A: Using the quadratic formula, the solutions are {sol2}."
            chunks.append({"text": q2 + a2, "source": "synthetic_quadratic"})
            
    return chunks

def generate_calculus_problems(count=1500):
    """Generate derivatives and integrals."""
    chunks = []
    x = sp.Symbol('x')
    print("[*] Generating Calculus Problems...")
    for _ in range(count):
        a = random.randint(1, 10)
        p = random.randint(2, 6)
        b = random.randint(1, 10)
        
        # Derivative
        expr = a * x**p + b * x
        deriv = sp.diff(expr, x)
        q1 = f"Q: What is the first derivative of f(x) = {expr} with respect to x?\n"
        a1 = f"A: Using the power rule, the derivative is f'(x) = {deriv}."
        chunks.append({"text": q1 + a1, "source": "synthetic_calculus_deriv"})
        
        # Integral
        integral = sp.integrate(expr, x)
        q2 = f"Q: Evaluate the indefinite integral of {expr} dx.\n"
        a2 = f"A: Applying the power rule for integration, the result is {integral} + C, where C is the constant of integration."
        chunks.append({"text": q2 + a2, "source": "synthetic_calculus_integ"})
        
    return chunks

def generate_theorem_qa():
    """Curated QA about Mathematical Theorems."""
    print("[*] Generating Theorem QA...")
    theorems = [
        ("Pythagorean Theorem", "In a right-angled triangle, the square of the hypotenuse side is equal to the sum of squares of the other two sides: a^2 + b^2 = c^2."),
        ("Fundamental Theorem of Calculus", "It links the concept of differentiating a function with the concept of integrating a function, showing they are inverse operations."),
        ("Fermat's Last Theorem", "No three positive integers a, b, and c satisfy the equation a^n + b^n = c^n for any integer value of n greater than 2."),
        ("Euler's Identity", "e^(i*pi) + 1 = 0, a beautiful equation connecting five fundamental mathematical constants."),
        ("Bolzano-Weierstrass Theorem", "Every bounded sequence in R^n has a convergent subsequence.")
    ]
    
    chunks = []
    for _ in range(500):
        for name, desc in theorems:
            q = f"Q: Can you explain the {name}?\nA: Yes, the {name} states that: {desc}"
            chunks.append({"text": q, "source": "synthetic_theorems"})
    return chunks

def main():
    final_data = []
    
    # 1. Gather Synthetic Data (Perfect equations and QA)
    final_data.extend(generate_algebra_problems(1500))
    final_data.extend(generate_calculus_problems(1500))
    final_data.extend(generate_theorem_qa())
    
    # 2. Gather PDF Descriptive Data
    for name, url in PDF_SOURCES.items():
        pdf_path = download_pdf(url, name)
        if pdf_path:
            pdf_text_chunks = extract_text_from_pdf(pdf_path, max_pages=200)
            final_data.extend(pdf_text_chunks)
            
    # Shuffle for better distribution during training
    random.shuffle(final_data)
    
    # Save dataset as JSONL
    train_file = os.path.join(DATA_DIR, "train.jsonl")
    eval_file = os.path.join(DATA_DIR, "eval.jsonl")
    
    split_idx = int(len(final_data) * 0.8)
    train_data = final_data[:split_idx]
    eval_data = final_data[split_idx:]
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(eval_file, "w", encoding="utf-8") as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
    print(f"\n[SUCCESS] Generated Math v2 Dataset")
    print(f"Train chunks: {len(train_data)} at {train_file}")
    print(f"Eval chunks: {len(eval_data)} at {eval_file}")
    print("Run `python train_cell.py --cell math_cell` to retrain!")

if __name__ == "__main__":
    main()
