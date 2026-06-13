import os
import json
import re
import time
import random
import requests
import wikipediaapi

QUALITY_FILTERS = {
    "min_tokens": 100,          # approximated via words
    "max_tokens": 512,          # approximated via words
    "min_alpha_ratio": 0.6,     # relaxed slightly for math/code
    "max_repetition_ratio": 0.3,
}

def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def chunk_text(text, max_words=350):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def passes_filters(chunk, cell_type):
    words = chunk.split()
    if len(words) < (QUALITY_FILTERS["min_tokens"] * 0.75): return False
    if len(words) > (QUALITY_FILTERS["max_tokens"] * 1.5): return False
    
    alpha_chars = sum(c.isalpha() for c in chunk)
    alpha_ratio = alpha_chars / max(len(chunk), 1)
    
    min_alpha = QUALITY_FILTERS["min_alpha_ratio"]
    if cell_type == "code_cell": min_alpha = 0.4 # Code has many symbols
    if cell_type == "math_cell": min_alpha = 0.5 # Math has numbers/symbols
    
    if alpha_ratio < min_alpha: return False
        
    vocab = set(words)
    rep_ratio = 1.0 - (len(vocab) / max(len(words), 1))
    if rep_ratio > QUALITY_FILTERS["max_repetition_ratio"]: return False
        
    return True

def save_data(chunks, cell_name):
    random.shuffle(chunks)
    split_idx = int(len(chunks) * 0.8)
    train_chunks = chunks[:split_idx]
    eval_chunks = chunks[split_idx:]
    
    os.makedirs(f"data/{cell_name}", exist_ok=True)
    
    with open(f"data/{cell_name}/train.jsonl", "w", encoding="utf-8") as f:
        for c in train_chunks: f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    with open(f"data/{cell_name}/eval.jsonl", "w", encoding="utf-8") as f:
        for c in eval_chunks: f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    print(f"[{cell_name}] Saved {len(train_chunks)} train chunks, {len(eval_chunks)} eval chunks.")

def prepare_history():
    print("Preparing History Data...")
    user_agent = 'CAMEL_MSLM_Research/1.0 (https://github.com/ALightbolt4G/CAMEL)'
    wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    topics = [
        "World War I", "World War II", "Treaty of Versailles", "League of Nations", 
        "Cold War", "French Revolution", "Industrial Revolution", "Roman Empire", 
        "Ancient Egypt", "American Civil War", "Ottoman Empire", "Byzantine Empire"
    ]
    
    chunks = []
    for topic in topics:
        page = wiki.page(topic)
        if not page.exists(): continue
        
        text = clean_text(page.text)
        page_chunks = chunk_text(text)
        for pc in page_chunks:
            if passes_filters(pc, "history_cell"):
                chunks.append({"text": pc, "source": f"Wikipedia: {topic}", "cell": "history_cell", "tokens": len(pc.split())})
        time.sleep(1) # Be nice to Wikipedia
        
    save_data(chunks, "history_cell")

def prepare_math():
    print("Preparing Math Data...")
    # Project Gutenberg raw TXT URLs for math books
    urls = [
        "https://www.gutenberg.org/cache/epub/33283/pg33283.txt", # Calculus Made Easy
        "https://www.gutenberg.org/files/26839/26839-0.txt",      # A First Book in Algebra
        "https://www.gutenberg.org/files/17384/17384-0.txt"       # Elements of Geometry
    ]
    chunks = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                text = clean_text(resp.text)
                text = text[text.find("*** START"):text.find("*** END")] if "*** START" in text else text
                page_chunks = chunk_text(text)
                for pc in page_chunks:
                    if passes_filters(pc, "math_cell"):
                        chunks.append({"text": pc, "source": f"Gutenberg: {url.split('/')[-1]}", "cell": "math_cell", "tokens": len(pc.split())})
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            
    save_data(chunks, "math_cell")

def prepare_code():
    print("Preparing Code Data...")
    pages = [
        "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/controlflow.rst",
        "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/datastructures.rst",
        "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/classes.rst",
        "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/modules.rst",
        "https://raw.githubusercontent.com/python/cpython/main/Doc/library/functions.rst",
        "https://raw.githubusercontent.com/python/cpython/main/Doc/reference/datamodel.rst"
    ]
    chunks = []
    for url in pages:
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                text = clean_text(resp.text)
                page_chunks = chunk_text(text)
                for pc in page_chunks:
                    if passes_filters(pc, "code_cell"):
                        chunks.append({"text": pc, "source": f"Python Docs: {url.split('/')[-1]}", "cell": "code_cell", "tokens": len(pc.split())})
        except Exception as e:
            print(f"Failed to fetch {url}: {e}")
            
    save_data(chunks, "code_cell")

if __name__ == "__main__":
    print("Starting CAMEL Data Preparation Pipeline...")
    os.makedirs("data", exist_ok=True)
    prepare_history()
    prepare_math()
    prepare_code()
    print("Pipeline Finished!")
