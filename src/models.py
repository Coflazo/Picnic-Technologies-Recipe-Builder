from __future__ import annotations

import json
import math
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, computed_field

# simple map to get everything down to grams, ml or pieces
UNIT_TO_BASE: dict[str, tuple[str, float]] = {
    "g": ("g", 1.0),
    "gram": ("g", 1.0),
    "grams": ("g", 1.0),
    "kg": ("g", 1000.0),
    "kilogram": ("g", 1000.0),
    "kilograms": ("g", 1000.0),
    "ml": ("ml", 1.0),
    "millilitre": ("ml", 1.0),
    "millilitres": ("ml", 1.0),
    "l": ("ml", 1000.0),
    "liter": ("ml", 1000.0),
    "litre": ("ml", 1000.0),
    "litres": ("ml", 1000.0),
    "liters": ("ml", 1000.0),
    "piece": ("piece", 1.0),
    "pieces": ("piece", 1.0),
    "pack": ("piece", 1.0),
    "packs": ("piece", 1.0),
}

def normalise_unit(value: float, unit: str) -> tuple[float, str]:
    key = unit.lower().strip()
    if key in UNIT_TO_BASE:
        base_unit, factor = UNIT_TO_BASE[key]
        return value * factor, base_unit
    return value, unit.lower()


class PriceTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Article(BaseModel):
    article_id: str = Field(alias="Article_ID")
    raw_name: str = Field(alias="Raw_Name")
    brand: str = Field(alias="Brand")
    price: float = Field(alias="Price")
    quantity_value: float = Field(alias="Quantity_Value")
    quantity_unit: str = Field(alias="Quantity_Unit")
    description: str = Field(alias="Description")
    price_per_unit: float = Field(alias="Price_Per_Unit")

    model_config = {"populate_by_name": True}

    @computed_field
    @property
    def quantity_base(self) -> float:
        val, _ = normalise_unit(self.quantity_value, self.quantity_unit)
        return val

    @computed_field
    @property
    def base_unit(self) -> str:
        _, u = normalise_unit(self.quantity_value, self.quantity_unit)
        return u

    @property
    def search_text(self) -> str:
        return f"{self.raw_name} {self.brand} {self.description}"

    @property
    def is_food(self) -> bool:
        # crude check to make sure we don't buy bleach for a cake recipe
        non_food_keywords = {
            "fabric softener", "dishwashing", "detergent", "laundry",
            "cleaning", "soap", "bleach", "sponge", "toilet",
        }
        text = f"{self.raw_name} {self.description}".lower()
        return not any(kw in text for kw in non_food_keywords)


class ParsedIngredient(BaseModel):
    name: str = Field()
    amount: float = Field()
    unit: str = Field()
    original_text: str = Field(default="")
    is_optional: bool = Field(default=False)


class ShoppingListItem(BaseModel):
    ingredient_name: str
    article: Article
    match_confidence: float = Field(ge=0.0, le=1.0)
    packs_needed: int = Field(ge=1)
    total_quantity: float
    total_quantity_unit: str
    total_price: float
    quantity_explanation: str = Field(default="")
    price_tier: PriceTier = PriceTier.MEDIUM
    is_optional: bool = False
    rerank_score: float = Field(default=0.0)


class ShoppingList(BaseModel):
    items: list[ShoppingListItem] = []
    total_cost: float = 0.0
    price_tier: PriceTier = PriceTier.MEDIUM
    recipe_text: str = ""
    parsed_ingredients_count: int = 0

    def compute_total(self) -> None:
        self.total_cost = round(sum(item.total_price for item in self.items), 2)


_DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "articles.json"


def load_catalog(path: Path | str | None = None) -> list[Article]:
    p = Path(path) if path else _DATA_FILE
    with open(p, "r", encoding="utf-8") as f:
        raw = json.load(f)
    articles = [Article.model_validate(item) for item in raw]
    return [a for a in articles if a.is_food]
