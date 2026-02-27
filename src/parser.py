import re
import math
from fractions import Fraction
from typing import Optional

from .models import ParsedIngredient, normalise_unit

def _parse_number(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.lower().strip()
    
    # fractions
    unicode_fracs = {"½": 0.5, "⅓": 1/3, "⅔": 2/3, "¼": 0.25, "¾": 0.75}
    for uf, val in unicode_fracs.items():
        if uf in text:
            prefix = text.replace(uf, "").strip()
            return (float(prefix) if prefix else 0.0) + val
    
    if "/" in text:
        try:
            return float(Fraction(text))
        except (ValueError, ZeroDivisionError):
            pass

    # text numbers
    word_nums = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "half": 0.5, "quarter": 0.25}
    if text in word_nums:
        return word_nums[text]

    try:
        return float(re.sub(r"[^\d\.]", "", text))
    except ValueError:
        return None

def _canonicalize(raw_entities: list[dict]) -> list[ParsedIngredient]:
    import logging
    logger = logging.getLogger("rich")
    
    parsed = []
    
    # physical rules mapping
    unit_map = {
        "g": ("g", 1.0), "gram": ("g", 1.0), "grams": ("g", 1.0),
        "kg": ("g", 1000.0), "kilogram": ("g", 1000.0),
        "ml": ("ml", 1.0), "milliliter": ("ml", 1.0), "milliliters": ("ml", 1.0),
        "l": ("ml", 1000.0), "liter": ("ml", 1000.0), "liters": ("ml", 1000.0),
        "pinch": ("g", 0.5), "pinches": ("g", 0.5),
        "clove": ("g", 5.0), "cloves": ("g", 5.0),
        "dash": ("ml", 1.0), "dashes": ("ml", 1.0),
        "tablespoon": ("ml", 15.0), "tbsp": ("ml", 15.0), "tablespoons": ("ml", 15.0),
        "teaspoon": ("g", 5.0), "tsp": ("g", 5.0), "teaspoons": ("g", 5.0),
        "cup": ("ml", 240.0), "cups": ("ml", 240.0),
        "handful": ("g", 30.0), "bunch": ("g", 30.0),
        "slice": ("piece", 1.0), "piece": ("piece", 1.0), "pieces": ("piece", 1.0),
    }

    for item in raw_entities:
        name = item.get("ingredient", "").strip()
        if not name or len(name) < 2:
            continue
            
        amt_raw = item.get("amount", "1")
        unit_raw = item.get("unit", "piece").lower()
        
        amt = _parse_number(amt_raw) or 1.0
        
        # normalize unitted math
        base_unit, factor = unit_map.get(unit_raw.rstrip('s'), ("piece", 1.0))
        if base_unit == "piece" and unit_raw not in unit_map:
            for k, v in unit_map.items():
                if k in unit_raw:
                    base_unit, factor = v
                    break
                    
        final_amount, final_base_unit = normalise_unit(amt * factor, base_unit)
        
        parsed.append(ParsedIngredient(
            name=name,
            amount=round(final_amount, 2),
            unit=final_base_unit,
            original_text=f"{amt_raw} {unit_raw} {name}".strip()
        ))
        
    return parsed

def parse_recipe(recipe_text: str, model) -> list[ParsedIngredient]:
    from rich.console import Console
    console = Console()
    
    # Tokenize and predict
    labels = ["ingredient", "amount", "unit"]
    entities = model.predict_entities(recipe_text, labels, threshold=0.3)
    
    # Sort them by occurrence
    entities = sorted(entities, key=lambda x: x["start"])
    
    # Group them dynamically based on proximity
    raw_dicts = []
    current_group = {}
    
    for ent in entities:
        label = ent["label"]
        text = ent["text"].lower()
        
        if label == "ingredient":
            if "ingredient" in current_group:
                # push previous
                raw_dicts.append(current_group)
                current_group = {}
            current_group["ingredient"] = text
        else:
            current_group[label] = text
            
        # if we have an ingredient, flush if we hit the next thing or at end (handled by loop)
        
    if current_group and "ingredient" in current_group:
        raw_dicts.append(current_group)

    # Some amounts/units might trail the ingredient, some precede. 
    # A cleaner approach for zero-shot outputting the exact list format requested.
    raw_dicts_cleaned = []
    
    # Let's do a sliding window group. Any amount/unit closest to the ingredient belongs to it.
    ing_nodes = [e for e in entities if e["label"] == "ingredient"]
    other_nodes = [e for e in entities if e["label"] != "ingredient"]
    
    for ing in ing_nodes:
        group = {"ingredient": ing["text"].lower()}
        
        # find closest amount
        amts = [o for o in other_nodes if o["label"] == "amount" and abs(o["start"] - ing["start"]) < 50]
        if amts:
            closest = min(amts, key=lambda o: abs(o["start"] - ing["start"]))
            group["amount"] = closest["text"]
            
        # find closest unit
        unis = [o for o in other_nodes if o["label"] == "unit" and abs(o["start"] - ing["start"]) < 50]
        if unis:
            closest = min(unis, key=lambda o: abs(o["start"] - ing["start"]))
            group["unit"] = closest["text"]
            
        raw_dicts_cleaned.append(group)

    return _canonicalize(raw_dicts_cleaned)
