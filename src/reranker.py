from __future__ import annotations

import math
from typing import Optional
import numpy as np
from .models import Article, ParsedIngredient, normalise_unit

# fake brand popularity scores to mimic what a real ML model would learn from purchase history
BRAND_POPULARITY: dict[str, float] = {
    "Picnic": 0.95, "Picnic House Brand": 0.90, "Picnic Organic": 0.85,
    "Picnic Bakery": 0.88, "Heinz": 0.80, "Campina": 0.78, "Blue Band": 0.72,
    "Calve": 0.76, "Chiquita": 0.70, "Dutch Green": 0.65, "Barilla": 0.75,
    "Mutti": 0.73, "Verstegen": 0.68, "Van Gilse": 0.66, "Jozo": 0.60,
    "Lassie": 0.64, "De Ruijter": 0.71, "Bonne Maman": 0.69, "Langnese": 0.62,
    "Hak": 0.63, "John West": 0.67, "Bonduelle": 0.65, "Stegeman": 0.61,
    "Conimex": 0.64, "Koh Thai": 0.60, "Maille": 0.66, "Zanetti": 0.68,
    "Dodoni": 0.63, "Maggi": 0.72, "Knorr": 0.74, "CoolBest": 0.67,
    "Appelsientje": 0.68, "Bar-le-Duc": 0.55, "Coca-Cola": 0.82,
    "Douwe Egberts": 0.80, "Pickwick": 0.70, "Milka": 0.78, "Lindt": 0.75,
    "Lays": 0.77, "Doritos": 0.76, "Duyvis": 0.68, "Al-Andalus": 0.55,
    "Rondeel": 0.62, "Robijn": 0.60, "Dreft": 0.58,
}

DEFAULT_POPULARITY = 0.50

def _name_overlap(ingredient_name: str, article_name: str) -> float:
    ing_tokens = set(ingredient_name.lower().split())
    art_tokens = set(article_name.lower().split())

    ing_tokens = {t for t in ing_tokens if len(t) > 2}
    art_tokens = {t for t in art_tokens if len(t) > 2}

    if not ing_tokens or not art_tokens: return 0.0
    
    # color matching penalty logic
    colors = {"red", "white", "yellow", "green", "black", "brown"}
    ing_colors = ing_tokens & colors
    art_colors = art_tokens & colors
    
    # if the user asked for a red onion, and the article is a white onion, that's a hard penalty
    color_penalty = 0.0
    if ing_colors and not (ing_colors & art_colors):
        color_penalty = 0.5 

    intersection = ing_tokens & art_tokens
    union = ing_tokens | art_tokens
    base_score = len(intersection) / len(union)
    return max(0.0, base_score - color_penalty)

def _excess_waste_ratio(required_amount: float, article: Article, packs: int) -> float:
    supplied = packs * article.quantity_base
    if required_amount <= 0: return 0.0
    return max(0.0, (supplied - required_amount) / required_amount)

def compute_features(
    ingredient: ParsedIngredient,
    article: Article,
    cosine_score: float,
    packs_needed: int,
) -> np.ndarray:
    f_cosine = cosine_score
    # Price filtering is strictly handled by the tier logic in solver.py now
    # We remove the continuous penalty here so premium items don't have artificially destroyed scores
    f_price = 1.0
    
    waste = _excess_waste_ratio(ingredient.amount, article, packs_needed)
    f_waste = 1.0 / (1.0 + waste)
    f_brand = BRAND_POPULARITY.get(article.brand, DEFAULT_POPULARITY)
    f_name = _name_overlap(ingredient.name, article.raw_name)

    return np.array([f_cosine, f_price, f_waste, f_brand, f_name], dtype=np.float32)

# weights that kinda act like a trained random forest
# order: [cosine_sim, price_penalty, waste_penalty, brand_popularity, name_overlap]
FEATURE_WEIGHTS = np.array([0.60, 0.05, 0.00, 0.10, 0.25], dtype=np.float32)

def rerank_score(
    ingredient: ParsedIngredient,
    article: Article,
    cosine_score: float,
    packs_needed: int,
) -> float:
    features = compute_features(ingredient, article, cosine_score, packs_needed)
    score = float(np.dot(features, FEATURE_WEIGHTS))
    return round(min(max(score, 0.0), 1.0), 4)

def rerank_candidates(
    ingredient: ParsedIngredient,
    candidates: list[tuple[Article, float, int]],
) -> list[tuple[Article, float, int, float]]:
    scored = []
    for article, cosine, packs in candidates:
        rs = rerank_score(ingredient, article, cosine, packs)
        scored.append((article, cosine, packs, rs))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored
