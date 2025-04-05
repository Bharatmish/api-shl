# combined_recommender.py

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from query_parser import parse_query_with_llm

# --- CONFIG ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAPPING_PATH = os.path.join(BASE_DIR, '..', 'embeddings', 'index_mapping.json')
FAISS_INDEX_PATH = os.path.join(BASE_DIR, '..', 'embeddings', 'faiss_index.bin')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- Load all assets ---
if not os.path.exists(MAPPING_PATH):
    raise FileNotFoundError(f"🔴 index_mapping.json not found at: {MAPPING_PATH}")

if not os.path.exists(FAISS_INDEX_PATH):
    raise FileNotFoundError(f"🔴 faiss_index.bin not found at: {FAISS_INDEX_PATH}")

with open(MAPPING_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

index = faiss.read_index(FAISS_INDEX_PATH)
model = SentenceTransformer(EMBEDDING_MODEL)

def score_item(item, parsed):
    score = 0

    # Type match
    if item["type"].lower() in [t.lower() for t in parsed["types"]]:
        score += 1.0

    # Duration match
    target_duration = parsed.get("duration")
    if target_duration:
        try:
            duration_val = int(item["duration"].split()[0])
            if duration_val <= target_duration:
                score += 1.0
        except:
            pass

    # Skill keyword match (title + description)
    for skill in parsed["skills"]:
        skill = skill.lower()
        if skill in item["name"].lower() or skill in item["description"].lower():
            score += 0.7

    # Bonus for remote/adaptive
    if item.get("remote") == "Yes":
        score += 0.2
    if item.get("adaptive") == "Yes":
        score += 0.2

    return score

# --- FINAL FUNCTION ---
def smart_recommend(query: str, top_k=10) -> list:
    print(f"\n🔍 Original query: {query}")
    parsed = parse_query_with_llm(query)
    print(f"\n📦 Parsed filters from LLM: {parsed}")

    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding), top_k * 3)  # Over-fetch more results

    candidates = []
    for i in I[0]:
        item = data[i]
        item_score = score_item(item, parsed)
        candidates.append((item_score, item))

    # Sort by score
    ranked = sorted(candidates, key=lambda x: x[0], reverse=True)
    top_items = [item for score, item in ranked[:top_k]]

    return top_items
