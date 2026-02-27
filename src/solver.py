from __future__ import annotations
import math
import logging
from typing import Optional
from .models import Article, ParsedIngredient, PriceTier, ShoppingListItem, normalise_unit

import numpy as np
from rich.console import Console
console = Console()

GARNISH_COST_THRESHOLD = 0.40
DEFAULT_MEAL_ESTIMATE = 15.0

_COMPATIBLE_UNITS = {
    "g": {"g"},
    "ml": {"ml"},
    "piece": {"piece"},
}

def _units_compatible(required_unit: str, article_unit: str) -> bool:
    req = required_unit.lower()
    art = article_unit.lower()
    if req == art: return True
    return art in _COMPATIBLE_UNITS.get(req, set())

def _calculate_packs(
    required_amount: float,
    article_quantity: float,
    required_unit: str,
    article_unit: str,
) -> Optional[tuple[int, float, str]]:
    art_qty, art_unit = normalise_unit(article_quantity, article_unit)

    if not _units_compatible(required_unit, art_unit):
        return None

    if art_unit == "piece":
        packs = max(1, math.ceil(required_amount / article_quantity))
        total = packs * article_quantity
        if packs == 1:
            explanation = f"1× pack ({total:.0f} {art_unit}s)"
        else:
            explanation = f"{packs}× packs ({total:.0f} {art_unit}s total, {required_amount:.0f} needed)"
        return packs, total, explanation

    packs = max(1, math.ceil(required_amount / art_qty))
    total = packs * art_qty

    if packs == 1:
        explanation = f"1× {article_quantity:.0f}{article_unit} ({total:.0f}{art_unit} total, {required_amount:.0f}{required_unit} needed)"
    else:
        explanation = f"{packs}× {article_quantity:.0f}{article_unit} ({total:.0f}{art_unit} total, {required_amount:.0f}{required_unit} needed)"

    return packs, total, explanation

def solve_ingredient(
    ingredient: ParsedIngredient,
    candidates: list[tuple[Article, float]],
    price_tier: PriceTier = PriceTier.MEDIUM,
    meal_cost_estimate: float = DEFAULT_MEAL_ESTIMATE,
) -> Optional[ShoppingListItem]:
    if not candidates:
        return None

    viable: list[tuple[Article, float, int, float, str]] = []

    for article, score in candidates:
        result = _calculate_packs(
            ingredient.amount,
            article.quantity_value,
            ingredient.unit,
            article.quantity_unit,
        )
        if result is None: continue
        packs, total_qty, explanation = result
        viable.append((article, score, packs, total_qty, explanation))

    if not viable:
        # User requested incompatible units (e.g. piece vs g)
        # Convert all to 1 pack so we can evaluate them.
        for article, score in candidates:
            packs = 1
            total_qty = getattr(article, "quantity_base", getattr(article, "quantity_value", 1.0))
            explanation = f"1× {article.quantity_value:.0f}{article.quantity_unit} (fallback semantic match)"
            viable.append((article, score, packs, total_qty, explanation))

    best_item = None
    best_waste = float("inf")
    best_score = -1.0

    for art, score, packs, total_qty, expl in viable:
        waste = total_qty - ingredient.amount
        total_price = packs * art.price

        is_garnish = (
            ingredient.is_optional
            or total_price > meal_cost_estimate * GARNISH_COST_THRESHOLD
            and ingredient.amount < 20 
        )

        if score > best_score or (score == best_score and waste < best_waste):
            best_waste = waste
            best_score = score
            best_item = ShoppingListItem(
                ingredient_name=ingredient.name,
                article=art,
                match_confidence=round(score, 4),
                packs_needed=packs,
                total_quantity=total_qty,
                total_quantity_unit=ingredient.unit,
                total_price=round(total_price, 2),
                quantity_explanation=expl,
                price_tier=price_tier,
                is_optional=is_garnish and ingredient.is_optional,
            )

    return best_item
