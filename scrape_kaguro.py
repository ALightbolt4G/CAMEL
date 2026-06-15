import os
import re
import json
import requests
import time
from bs4 import BeautifulSoup
import random

DATA_DIR = "data/rezero_cell"
TRAIN_PATH = os.path.join(DATA_DIR, "train.jsonl")
EVAL_PATH = os.path.join(DATA_DIR, "eval.jsonl")

def clean_novel_text(text):
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('※', '').replace('', "'").strip()
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

print("Scraping KaguroJP for Arc 3...")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

toc_url = "https://kagurojp.wordpress.com/table-of-contents/"
r = requests.get(toc_url, headers=headers, timeout=10)
soup = BeautifulSoup(r.text, 'html.parser')

links = []
for a in soup.find_all('a', href=True):
    href = a['href']
    if 'vol-3-ch-' in href and href not in links:
        links.append(href)

print(f"Found {len(links)} chapter links for Arc 3 on KaguroJP.")

novel_chunks = []

for link in links:
    print(f"Scraping chapter: {link}")
    try:
        r = requests.get(link, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        content_div = soup.find('div', {'class': 'entry-content'})
        if not content_div:
            continue
            
        paragraphs = content_div.find_all('p')
        chapter_text = ""
        
        for p in paragraphs:
            text = clean_novel_text(p.text)
            if len(text) > 30 and "Previous Chapter" not in text and "Next Chapter" not in text and "Translation" not in text:
                chapter_text += text + " "
                
        words = chapter_text.split()
        chunk_size = 100
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            if len(chunk.split()) >= 50:
                # Add arc-3 tag to source so it gets counted properly
                novel_chunks.append(create_chunk(chunk, "novel", source=link + "#arc-3"))
                
        time.sleep(1)
    except Exception as e:
        print(f"Failed to scrape {link}: {e}")

print(f"\nScraped {len(novel_chunks)} new novel chunks from KaguroJP.")

# Combine all
all_chunks = all_chunks + novel_chunks
random.shuffle(all_chunks)

total = len(all_chunks)
train_size = int(total * 0.8)

train_data = all_chunks[:train_size]
eval_data = all_chunks[train_size:]

print(f"Writing {train_size} training chunks...")
with open(TRAIN_PATH, "w", encoding="utf-8") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Writing {len(eval_data)} eval chunks...")
with open(EVAL_PATH, "w", encoding="utf-8") as f:
    for item in eval_data:
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
print("FINAL DATASET COMPLETE (ADDED KAGUROJP)")
print("="*50)
print(f"Arc 1: {arcs['arc_1']} chunks")
print(f"Arc 2: {arcs['arc_2']} chunks")
print(f"Arc 3: {arcs['arc_3']} chunks")
print(f"Arc 4: {arcs['arc_4']} chunks")
print(f"Arc 5: {arcs['arc_5']} chunks")
print(f"Arc 6: {arcs['arc_6']} chunks")
print("="*50)
