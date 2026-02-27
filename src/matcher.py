# matches ingredients to articles using onnxruntime and faiss in bulk

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

import numpy as np

from .models import Article, load_catalog

logger = logging.getLogger(__name__)

# where we keep the faiss stuff
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_INDEX_DIR = _PROJECT_ROOT / "data" / "index"
_INDEX_FILE = _INDEX_DIR / "faiss.index"
_ARTICLE_IDS_FILE = _INDEX_DIR / "article_ids.json"
_ONNX_MODEL_DIR = _PROJECT_ROOT / "onnx_model"

# globals for the model so we don't reload it

class ArticleMatcher:
    def __init__(self, global_state: dict):
        self._index = global_state.get("faiss_index")
        self._articles = global_state.get("articles", [])
        self._article_map = global_state.get("article_map", {})
        self._article_ids = global_state.get("article_ids", [])
        self._tokenizer = global_state.get("tokenizer")
        self._model = global_state.get("onnx_model")
        self._vec_cache: dict[str, np.ndarray] = {}

    @lru_cache(maxsize=5000)
    def _generate_single_embedding_lru(self, ingredient_name: str) -> tuple:
        import faiss
        inputs = self._tokenizer(ingredient_name, return_tensors="np", padding=True, truncation=True)
        ort_inputs = {k: v for k, v in inputs.items()}
        outputs = self._model.run(None, ort_inputs)
        last_hidden_state = outputs[0]
        
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = np.expand_dims(attention_mask, -1)
        sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
        vector = (sum_embeddings / sum_mask).astype(np.float32)
        
        faiss.normalize_L2(vector)
        return tuple(vector[0].tolist())

    def _get_batch_embeddings(self, ingredient_names: list[str]) -> np.ndarray:
        import faiss
        
        uncached_names = [name for name in ingredient_names if name not in self._vec_cache]
        
        if uncached_names:
            inputs = self._tokenizer(uncached_names, return_tensors="np", padding=True, truncation=True)
            ort_inputs = {k: v for k, v in inputs.items()}
            outputs = self._model.run(None, ort_inputs)
            last_hidden_state = outputs[0]
            
            # mean pooling
            attention_mask = inputs["attention_mask"]
            input_mask_expanded = np.expand_dims(attention_mask, -1)
            sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
            vectors = (sum_embeddings / sum_mask).astype(np.float32)
            
            faiss.normalize_L2(vectors)
            
            for name, vec in zip(uncached_names, vectors):
                self._vec_cache[name] = vec
                
        final_matrix = np.array([self._vec_cache[name] for name in ingredient_names])
        return final_matrix

    def match_bulk(
        self,
        ingredient_names: list[str],
        top_k: int = 10,
        min_score: float = 0.45,
    ) -> dict[str, list[tuple[Article, float]]]:
        if not ingredient_names:
            return {}

        query_matrix = self._get_batch_embeddings(ingredient_names)
        
        distances, indices = self._index.search(query_matrix, top_k)
        
        results_map = {}
        # specific fragile items that struggle with vector representation
        fragile_keywords = {"mint", "tomato", "parsley", "basil", "coriander", "cilantro", "thyme", "rosemary"}
        
        for i, name in enumerate(ingredient_names):
            matches = []
            
            # dynamically adjust threshold
            adaptive_min = min_score
            if any(k in name.lower() for k in fragile_keywords):
                adaptive_min = 0.35
                
            for score, idx in zip(distances[i], indices[i]):
                if idx < 0 or idx >= len(self._article_ids):
                    continue
                if score < adaptive_min:
                    continue
                
                article_id = self._article_ids[idx]
                article = self._article_map.get(article_id)
                if article:
                    matches.append((article, float(score)))
            results_map[name] = matches
            
        return results_map
