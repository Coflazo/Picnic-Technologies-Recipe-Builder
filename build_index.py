#!/usr/bin/env python3

import json
import faiss
import numpy as np
from pathlib import Path
from src.models import load_catalog
from transformers import AutoTokenizer
import onnxruntime as ort

def main():
    print("Loading catalog...")
    articles = load_catalog()
    article_names = [a.raw_name for a in articles]
    article_ids = [a.article_id for a in articles]
    print(f"Found {len(articles)} food items.")

    print("Loading ONNX models...")
    tokenizer = AutoTokenizer.from_pretrained("onnx_model", local_files_only=True)
    model = ort.InferenceSession("onnx_model/model.onnx", providers=['CPUExecutionProvider'])
    
    print("Computing 384d semantic vectors...")
    batch_size = 50
    all_vectors = []
    
    for i in range(0, len(article_names), batch_size):
        batch_names = article_names[i:i+batch_size]
        inputs = tokenizer(batch_names, return_tensors="np", padding=True, truncation=True)
        ort_inputs = {k: v for k, v in inputs.items()}
        outputs = model.run(None, ort_inputs)
        last_hidden_state = outputs[0]
        
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        vectors = (sum_embeddings / sum_mask).astype(np.float32)
        faiss.normalize_L2(vectors)
        all_vectors.append(vectors)
        
    final_matrix = np.vstack(all_vectors)
    
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(384)
    index.add(final_matrix)
    
    _PROJECT_ROOT = Path(__file__).resolve().parent
    faiss.write_index(index, str(_PROJECT_ROOT / "data" / "index" / "faiss.index"))
    
    print("Saving Article IDs map...")
    with open(_PROJECT_ROOT / "data" / "index" / "article_ids.json", "w") as f:
        json.dump(article_ids, f)

    print("OK! Database generated and linked.")

if __name__ == "__main__":
    main()
