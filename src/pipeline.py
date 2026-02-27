# glues everything together: text -> parser -> matcher -> solver -> reranker -> list

from __future__ import annotations

import logging
import time
from typing import Optional, Any

from .matcher import ArticleMatcher
from .models import (
    Article,
    ParsedIngredient,
    PriceTier,
    ShoppingList,
    ShoppingListItem,
    load_catalog,
)
from .parser import parse_recipe
from .solver import solve_ingredient

logger = logging.getLogger(__name__)

def apply_price_tier(candidates: list[Any], tier: str) -> dict | None:
    """
    Filters candidates using a semantic guardrail (90% threshold),
    then sorts by unit price based on the selected tier.
    """
    if not candidates:
        return None

    # Safely normalize the candidates to a list of dicts regardless of FAISS output type
    normalized_candidates = []
    for c in candidates:
        if isinstance(c, tuple) and len(c) == 2:
            normalized_candidates.append({"article": c[0], "score": c[1]})
        elif isinstance(c, dict):
            normalized_candidates.append(c)

    if not normalized_candidates:
        return None

    # Step 1: Semantic Guardrail
    max_score = normalized_candidates[0]["score"]
    valid_candidates = [c for c in normalized_candidates if c["score"] >= (max_score * 0.90)]
    
    # Step 2: Tiered Execution
    tier_lower = tier.lower()
    if tier_lower == "low":
        # Return the absolute cheapest item per unit in the valid pool
        return min(valid_candidates, key=lambda x: x["article"].price_per_unit)
    
    elif tier_lower == "high":
        # Return the absolute most expensive item per unit in the valid pool
        return max(valid_candidates, key=lambda x: x["article"].price_per_unit)
    
    else:
        # Medium tier: Stick to the original best semantic match
        return valid_candidates[0]


class RecipeBuilderPipeline:
    def __init__(self, global_state: dict):
        self._matcher = ArticleMatcher(global_state)
        self._gliner = global_state.get("gliner")

    def parse(self, recipe_text: str) -> list[ParsedIngredient]:
        return parse_recipe(recipe_text, self._gliner)

    def generate_shopping_list(
        self,
        recipe_text: str,
        price_tier: PriceTier = PriceTier.MEDIUM,
        top_k: int = 10,
    ) -> ShoppingList:

        from rich.console import Console
        console = Console()
        
        console.print(f"\n[bold #eeeeee]New Recipe Request[/bold #eeeeee] (Tier: [#198917]{price_tier.value.upper()}[/#198917])")
        
        t0 = time.perf_counter()
        
        # 1. NLP Extraction
        ingredients = self.parse(recipe_text)
        t_nlp = (time.perf_counter() - t0) * 1000
        
        items: list[ShoppingListItem] = []

        if not ingredients:
            console.print(f"  [#ea4b4b]x[/#ea4b4b] [#eeeeee]NLP Zero-Shot Extraction: 0 items ({t_nlp:.1f}ms)[/#eeeeee]")
            return ShoppingList(
                items=[],
                price_tier=price_tier,
                recipe_text=recipe_text,
                parsed_ingredients_count=0
            )

        console.print(f"  [#198917]OK[/#198917] [#eeeeee]NLP Zero-Shot Extraction: {len(ingredients)} items ({t_nlp:.1f}ms)[/#eeeeee]")

        # 2. Batched Vector Search
        ingredient_names = [ing.name for ing in ingredients]
        
        t1 = time.perf_counter()
        bulk_matches = self._matcher.match_bulk(ingredient_names, top_k=top_k)
        t_faiss = (time.perf_counter() - t1) * 1000
        console.print(f"  [#198917]OK[/#198917] [#eeeeee]FAISS Vector Encoding & Match: ({t_faiss:.1f}ms)[/#eeeeee]")

        t2 = time.perf_counter()

        # Check if match_bulk returned a dict or a list to iterate safely
        is_dict_matches = isinstance(bulk_matches, dict)

        # 3. Apply Tiering and Constraints
        for i, ingredient in enumerate(ingredients):
            if is_dict_matches:
                raw_candidates = bulk_matches.get(ingredient.name, [])
            else:
                raw_candidates = bulk_matches[i] if i < len(bulk_matches) else []

            if not raw_candidates:
                continue

            # Pass the raw candidates to the tiering algorithm
            best_match = apply_price_tier(raw_candidates, price_tier.value)

            if best_match:
                selected_article = best_match["article"]
                selected_score = best_match["score"]
                
                # Hand off the finalized selection to the solver
                item = solve_ingredient(
                    ingredient,
                    [(selected_article, selected_score)],
                    price_tier=price_tier,
                )

                if item:
                    item.rerank_score = selected_score
                    items.append(item)

        t_solver = (time.perf_counter() - t2) * 1000
        console.print(f"  [#198917]OK[/#198917] [#eeeeee]Neural Re-ranking & Constraints: ({t_solver:.1f}ms)[/#eeeeee]")

        console.print("\n  [bold #eeeeee]Matches:[/bold #eeeeee]")
        total_cost = 0.0
        for item in items:
            prefix = "[dim]opt[/dim]" if item.is_optional else "[#198917]req[/#198917]"
            console.print(f"  - ({prefix}) [#198917]{item.ingredient_name}[/#198917]: [#eeeeee]{item.article.raw_name} ({item.article.brand}) - €{item.total_price:.2f}[/#eeeeee]")
            total_cost += item.total_price

        t_total = (time.perf_counter() - t0) * 1000
        console.print(f"\n  [bold #eeeeee]Total Execution Time:[/bold #eeeeee] [#198917]{t_total:.1f}ms[/#198917]\n")

        shopping_list = ShoppingList(
            items=items,
            price_tier=price_tier,
            recipe_text=recipe_text,
            parsed_ingredients_count=len(ingredients),
        )
        
        # Ensure total is strictly calculated from the final items
        shopping_list.total_cost = round(total_cost, 2)
        if hasattr(shopping_list, 'compute_total'):
            shopping_list.compute_total()

        return shopping_list
