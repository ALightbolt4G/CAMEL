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
    # Remove weird characters that might be in the translation
    text = text.replace('※', '').strip()
    return text

def create_chunk(text, chunk_type, source=""):
    return {
        "text": text,
        "type": chunk_type,
        "cell": "rezero_cell",
        "source": source
    }

print("Loading existing dataset...")
existing_chunks = []
if os.path.exists(TRAIN_PATH):
    with open(TRAIN_PATH, 'r', encoding='utf-8') as f:
        existing_chunks.extend([json.loads(line) for line in f])
if os.path.exists(EVAL_PATH):
    with open(EVAL_PATH, 'r', encoding='utf-8') as f:
        existing_chunks.extend([json.loads(line) for line in f])

# Filter out the fake novel chunks
filtered_chunks = [c for c in existing_chunks if c['type'] != 'novel']
print(f"Removed {len(existing_chunks) - len(filtered_chunks)} fake novel chunks.")

print("Scraping WCT Arc 1-6 chapters...")
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

arc_targets = {
    "arc-1": 3,
    "arc-2": 3,
    "arc-3": 5,
    "arc-4": 8,
    "arc-5": 3,
    "arc-6": 5,
}

novel_chunks = []

for arc_key, target_count in arc_targets.items():
    print(f"\nFetching links for {arc_key}...")
    arc_url = f"https://witchculttranslation.com/{arc_key}/"
    chapter_links = []
    try:
        r = requests.get(arc_url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, 'html.parser')
        content = soup.find('div', {'class': 'entry-content'})
        if content:
            for a in content.find_all('a', href=True):
                href = a['href']
                if ('arc-' in href or 'chapter' in href) and href not in chapter_links and 'witchculttranslation' in href:
                    chapter_links.append(href)
        time.sleep(1)
    except Exception as e:
        print(f"Failed to fetch {arc_url}: {e}")
        continue
        
    print(f"Found {len(chapter_links)} chapter links for {arc_key}.")
    
    # Shuffle only within this arc's links to randomize chapters, but limit to target_count
    random.shuffle(chapter_links)
    selected_links = chapter_links[:target_count]
    
    for link in selected_links:
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
                if len(text) > 30 and "Previous Chapter" not in text and "Next Chapter" not in text:
                    chapter_text += text + " "
                    
            words = chapter_text.split()
            chunk_size = 100
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if len(chunk.split()) >= 50:
                    novel_chunks.append(create_chunk(chunk, "novel", source=link))
                    
            time.sleep(1)
        except Exception as e:
            print(f"Failed to scrape {link}: {e}")

print(f"\nScraped {len(novel_chunks)} real novel chunks.")

# Combine and save
all_chunks = filtered_chunks + novel_chunks
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

print("\n" + "="*50)
print("REAL DATASET UPDATE COMPLETE")
print("="*50)
print(f"Total Chunks: {total}")
print(f"Descriptive (Wiki): {sum(1 for c in all_chunks if c['type'] == 'descriptive')} ({(sum(1 for c in all_chunks if c['type'] == 'descriptive')/total)*100:.1f}%)")
print(f"Q&A: {sum(1 for c in all_chunks if c['type'] == 'qa')} ({(sum(1 for c in all_chunks if c['type'] == 'qa')/total)*100:.1f}%)")
print(f"Real Novel Text: {len(novel_chunks)} ({(len(novel_chunks)/total)*100:.1f}%)")
print("="*50)
