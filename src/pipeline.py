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
from .reranker import rerank_candidates
from .solver import solve_ingredient, _calculate_packs

logger = logging.getLogger(__name__)

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
            candidates = bulk_matches.get(ingredient.name, [])

            if not candidates:
                continue

            # Step 1: Semantic Guardrail (Dynamic Thresholding)
            s_max = candidates[0][1]
            valid_candidates = [c for c in candidates if c[1] >= 0.90 * s_max]

            # Step 2: The Tiered Execution Logic
            if price_tier == PriceTier.LOW:
                # Bypass re-ranker, grab absolute minimum price
                best_candidate = min(valid_candidates, key=lambda x: x[0].price_per_unit)
                item = solve_ingredient(ingredient, [best_candidate], price_tier=price_tier)
                if item:
                    item.rerank_score = best_candidate[1]
                    items.append(item)
                    
            elif price_tier == PriceTier.HIGH:
                # Bypass re-ranker, grab absolute maximum price
                best_candidate = max(valid_candidates, key=lambda x: x[0].price_per_unit)
                item = solve_ingredient(ingredient, [best_candidate], price_tier=price_tier)
                if item:
                    item.rerank_score = best_candidate[1]
                    items.append(item)
                    
            else:
                # MEDIUM: Run the Multi-Criteria Re-Ranker formula
                candidates_with_packs = []
                for article, score in valid_candidates:
                    result = _calculate_packs(
                        ingredient.amount,
                        article.quantity_value,
                        ingredient.unit,
                        article.quantity_unit,
                    )
                    if result:
                        packs, _, _ = result
                        candidates_with_packs.append((article, score, packs))
                    else:
                        # if the math fails, just buy 1
                        candidates_with_packs.append((article, score, 1))

                # sort them so the good options bubble to the top
                reranked = rerank_candidates(ingredient, candidates_with_packs)

                # pick the best one that fits our budget
                reranked_as_candidates = [(art, cos) for art, cos, _, _ in reranked]
                item = solve_ingredient(
                    ingredient,
                    reranked_as_candidates,
                    price_tier=price_tier,
                )

                if item:
                    if reranked:
                        item.rerank_score = reranked[0][3]
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
