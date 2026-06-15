import os
import json
import requests
import time
import re
import io
import PyPDF2

DATA_DIR = "data/rezero_cell"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "eval.jsonl")

def clean_novel_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('※', '').replace(chr(39), '').strip()
    return text

def create_chunk(text, chunk_type, source=""):
    return {
        "text": text,
        "type": chunk_type,
        "cell": "rezero_cell",
        "source": source
    }

print("Loading existing dataset...")
all_chunks = []
if os.path.exists(TRAIN_PATH):
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        all_chunks.extend([json.loads(line) for line in f])
if os.path.exists(EVAL_PATH):
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        all_chunks.extend([json.loads(line) for line in f])

print("Scraping WCT Arc 4 PDF...")
url = 'https://witchculttranslation.com/wp-content/uploads/2018/11/a4-ph01-c001-c023-unrevised.pdf?x20762'
headers = {'User-Agent': 'Mozilla/5.0'}

novel_chunks = []
try:
    r = requests.get(url, headers=headers)
    pdf = PyPDF2.PdfReader(io.BytesIO(r.content))
    print(f"PDF loaded. Pages: {len(pdf.pages)}")
    
    chapter_text = ""
    # Process up to 50 pages to get enough chunks
    for i in range(min(50, len(pdf.pages))):
        text = pdf.pages[i].extract_text()
        if text:
            chapter_text += clean_novel_text(text) + " "
            
    words = chapter_text.split()
    chunk_size = 100
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        if len(chunk.split()) >= 50:
            novel_chunks.append(create_chunk(chunk, "novel", source=url + f"#arc-4"))
            
except Exception as e:
    print('Failed:', e)

print(f"\nScraped {len(novel_chunks)} new novel chunks for Arc 4.")

import random
all_chunks = all_chunks + novel_chunks
random.shuffle(all_chunks)

total = len(all_chunks)
train_size = int(total * 0.8)

with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    for item in all_chunks[:train_size]:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open(EVAL_PATH, "w", encoding="utf-8") as f:
    for item in all_chunks[train_size:]:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# Calculate totals
arcs = {f"arc_{i}": 0 for i in range(1, 7)}
for c in all_chunks:
    if c.get("type") == "novel":
        src = c.get("source", "")
        for i in range(1, 7):
            if f"arc-{i}" in src or f"arc_{i}" in src:
                arcs[f"arc_{i}"] += 1

print("\n" + "="*50)
print("FINAL DATASET COMPLETE (ADDED ARC 4 PDF)")
print("="*50)
print(f"Arc 1: {arcs['arc_1']} chunks")
print(f"Arc 2: {arcs['arc_2']} chunks")
print(f"Arc 3: {arcs['arc_3']} chunks")
print(f"Arc 4: {arcs['arc_4']} chunks")
print(f"Arc 5: {arcs['arc_5']} chunks")
print(f"Arc 6: {arcs['arc_6']} chunks")
print("="*50)
