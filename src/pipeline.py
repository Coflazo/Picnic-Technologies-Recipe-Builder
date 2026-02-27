# glues everything together: text -> parser -> matcher -> solver -> reranker -> list

from __future__ import annotations

import logging
import time
from typing import Optional

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

def apply_price_tier(candidates: list[dict], tier: str) -> dict | None:
    if not candidates:
        return None

    # Step 1: Semantic Guardrail
    # Get the cosine similarity score of the absolute best mathematical match
    max_score = candidates[0]["score"]
    
    # Keep only items that are at least 90 percent as accurate as the top match
    valid_candidates = [c for c in candidates if c["score"] >= (max_score * 0.90)]
    
    # Step 2: Tiered Execution
    tier_lower = tier.lower()
    if tier_lower == "low":
        # Return the absolute cheapest item per unit in the valid pool
        return min(valid_candidates, key=lambda x: x["article"].price_per_unit)
    
    elif tier_lower == "high":
        # Return the absolute most expensive item per unit in the valid pool
        return max(valid_candidates, key=lambda x: x["article"].price_per_unit)
    
    else:
        # Medium tier: Stick to the original best semantic match (index 0)
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
        # extract stuff from the text
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

        # grab all names and do one giant fast matrix lookup 
        ingredient_names = [ing.name for ing in ingredients]
        
        t1 = time.perf_counter()
        bulk_matches = self._matcher.match_bulk(ingredient_names, top_k=top_k)
        t_faiss = (time.perf_counter() - t1) * 1000
        console.print(f"  [#198917]OK[/#198917] [#eeeeee]FAISS Vector Encoding & Match: ({t_faiss:.1f}ms)[/#eeeeee]")

        t2 = time.perf_counter()

        for ingredient in ingredients:
            raw_candidates = bulk_matches.get(ingredient.name, [])

            if not raw_candidates:
                continue

            # Convert to dict format for the Tiered Bracket Algorithm
            matches = [{"article": art, "score": score} for art, score in raw_candidates]

            # Pass the list of FAISS matches AND the requested price tier to the new function
            best_match = apply_price_tier(matches, price_tier.value)

            if best_match:
                selected_article = best_match["article"]
                selected_score = best_match["score"]
                
                # Pass the single selected article to the solver to calculate exact packs needed
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
        for item in items:
            prefix = "[dim]opt[/dim]" if item.is_optional else "[#198917]req[/#198917]"
            console.print(f"  - ({prefix}) [#198917]{item.ingredient_name}[/#198917]: [#eeeeee]{item.article.raw_name} ({item.article.brand}) - €{item.total_price:.2f}[/#eeeeee]")

        t_total = (time.perf_counter() - t0) * 1000
        console.print(f"\n  [bold #eeeeee]Total Execution Time:[/bold #eeeeee] [#198917]{t_total:.1f}ms[/#198917]\n")

        shopping_list = ShoppingList(
            items=items,
            price_tier=price_tier,
            recipe_text=recipe_text,
            parsed_ingredients_count=len(ingredients),
        )
        shopping_list.compute_total()

        return shopping_list
