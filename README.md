<p align="center">
  <img src="https://images.ctfassets.net/8vofjvai1hpv/6kPVsBovtEASHsV9zTvq5n/09ccafc8e3bf48186cf37927fdaa37c2/logo-Picnic.png?w=525&h=225&q=90&fm=png&bg=transparent" alt="Picnic Logo" width="300" />
</p>

<h1 align="center">Picnic Technologies Recipe Builder PoC</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/FAISS-000000?style=flat-square&logoColor=white" alt="FAISS" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face" />
</p>

## Acknowledgments & Context

This project was born out of a Picnic Technologies in-house day. During discussions, the team mentioned they were actively working on an AI feature to parse user-generated recipes and automatically add the correct items to the shopping basket. This repository serves as a focused Proof of Concept (PoC) exploring low-latency vector retrieval and dynamic economic constraints to solve that exact challenge. 

A sincere thank you to the Picnic Technologies engineering team, and specifically to Maya Budhdeo, for providing the technical insights, context, and the incredible opportunity to explore this architecture. 

## The Problem Statement

Bridging unstructured human text (recipes) with structured e-commerce databases is historically difficult. Recipes contain massive semantic drift, varied and informal measurements, and absolutely no concept of brand tiering. A human reading "a bunch of carrots" intuitively knows to buy a standard 500g bag, but traditional strict keyword-matching systems will fail utterly if the catalog only lists "Washed Snack Carrots". 

Furthermore, recommending products is not just about semantic accuracy. It requires economic constraints. A user asking for "avocado" needs a system capable of differentiating between a bulk budget net, a standard ripe unit, and a premium organic variant based upon dynamic pricing tiers.

## System Architecture

> Objective: Sub-600ms API inference times mapping unstructured text to a financially constrained basket.

### Client-Server Flow

```mermaid
sequenceDiagram
    participant User as Frontend Client
    participant API as FastAPI Backend
    participant NLP as GLiNER Model
    participant Engine as FAISS Vector DB
    participant Alg as Tiered Bracket Algorithm

    User->>API: POST /api/shopping-list (Recipe Text, "LOW" Tier)
    API->>NLP: Extract Ingredients (Zero-Shot NER)
    NLP-->>API: Parsed Objects (Name, Amount, Unit)
    API->>Engine: Batched ONNX Vectorization (384d semantic encoding)
    Engine-->>API: Top 10 Candidate Matches per Item
    API->>Alg: Apply Semantic Bounding & Price Filters
    Alg-->>API: Optimal Constrained Replacements
    API-->>User: JSON Shopping Cart (Total Price, Exact Packs)
```

### Internal Processing Pipeline

```mermaid
flowchart TD
    A[Raw Unstructured Text] --> B(GLiNER Zero-Shot NER)
    B --> C{Unit Canonicalization}
    C --> D[SentenceTransformers Batched Vectorization]
    D --> E[FAISS Index High-Dimensional Search]
    E --> F[Semantic Bounding Box Filter]
    F --> G[Tiered Bracket Algorithm]
    G --> H((Multi-Criteria Re-Ranker))
    H --> I[JSON Output Cart]
```

## Technical Deep Dive & Justifications

### FastAPI with Lifespan Context
Machine learning models (PyTorch tensors, ONNX pipelines) and vector databases (FAISS) are exceptionally heavy to load. By trapping these initializations purely within the FastAPI `lifespan` sequence, the models reside persistently in system RAM (or VRAM). This eliminates the devastating cold-start penalty, allowing real-time HTTP requests to achieve sub-600ms latency without asynchronous locking.

### GLiNER (gliner_small-v2.1)
Traditional NLP layers like SpaCy require rigid fine-tuning to detect edge-case culinary entities. In contrast, routing every request to a massive cloud LLM introduces unacceptable latency and API costs. GLiNER provides a perfect middle ground: lightweight, local, zero-shot Named Entity Recognition. It autonomously extracts and cleans ingredients, quantities, and fractional measurements instantly on the CPU.

### SentenceTransformers & FAISS
To solve the semantic drift problem, string matching was completely abandoned. Instead, the catalog is mapped utilizing the `all-MiniLM-L6-v2` transformer. Every catalog item is translated into a 384-dimensional mathematical vector. 

At runtime, user ingredients are vectorized and mapped against the index using FAISS (Facebook AI Similarity Search). This allows the execution of highly optimized C++ matrix searches against the catalog, meaning the AI natively understands that "Iceberg Lettuce" is an almost identical mathematical proximity to "Romaine", silently ensuring smart substitutions rather than failing the request.

### The Semantic-Bounded Price Sort (Tiered Bracket Algorithm)
Handling the "Budget Tier" requirement is complex. If a system simply sorts the FAISS results by price, a request for "Chicken Breast" might return "Chicken Bouillon Cubes" purely because they are cheaper and share the word "chicken".

> **Note**: The budget tier logic is currently a lightweight proof-of-concept. It was implemented as a fun, experimental idea to explore economic constraints alongside semantic search. It might not work perfectly in every edge case yet, but expect more robust refinements in future updates! ;)

To resolve this, the pipeline executes a Semantic Bounding Box. It extracts the absolute highest cosine similarity score from the FAISS candidates ($S_{max}$). It then immediately drops any candidate whose score falls below 90% of that top match. 

This creates a highly-filtered, semantically pristine pool of essentially identical products (e.g., Budget Avocado, Standard Avocado, Premium Avocado). 
For the `LOW` and `HIGH` tiers, the Re-Ranker is completely bypassed. The system runs a raw `argmin` or `argmax` over the bounded pool unit prices to ensure absolute adherence to the requested economic constraint without sacrificing semantic accuracy.

```python
def apply_price_tier(candidates: list[dict], tier: str) -> dict:
    if not candidates:
        return None

    # Step 1: The Semantic Guardrail
    # Grab the cosine similarity score of the absolute best match
    max_score = candidates[0]["score"]
    
    # Keep only items that are at least 90 percent as accurate as the top match
    valid_candidates = [c for c in candidates if c["score"] >= (max_score * 0.90)]
    
    # Step 2: Tiered Execution
    tier = tier.lower()
    if tier == "low":
        # Find the absolute cheapest item per unit in the valid pool
        return min(valid_candidates, key=lambda x: x["article"].price_per_unit)
    
    elif tier == "high":
        # Find the absolute most expensive item per unit in the valid pool
        return max(valid_candidates, key=lambda x: x["article"].price_per_unit)
    
    else:
        # Medium tier: Stick to the original best semantic/popular match
        return valid_candidates[0]
```

## Project Structure

```text
├── .venv/
├── build_index.py
├── data/
│   ├── articles.json
│   └── index/
│       ├── article_ids.json
│       └── faiss.index
├── frontend/
│   ├── index.html
│   ├── main.js
│   └── style.css
├── onnx_model/
├── pyproject.toml
├── run.py
└── src/
    ├── __init__.py
    ├── api.py
    ├── matcher.py
    ├── models.py
    ├── parser.py
    ├── pipeline.py
    ├── reranker.py
    └── solver.py
```

## Getting Started

### Prerequisites
Ensure you have Python 3.11+ installed. 

### Installation

1. Clone the repository block:
```bash
git clone https://github.com/Coflazo/Picnic-Technologies-Recipe-Builder.git
cd Picnic-Technologies-Recipe-Builder
```

2. Create and activate a pristine Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install the dependencies utilizing `pip`:
```bash
pip install -r requirements.txt
```

> Note: If building the FAISS index on macOS, ensure you prefix your scripts with `TOKENIZERS_PARALLELISM=false` to prevent internal process deadlocks from Hugging Face's tokenizer module.

### Updating the Catalog (articles.json)
If you add, remove, or modify any items within `data/articles.json`, the FAISS vector database will go out of sync (the "Ghost Index" issue). 

To synchronize the AI with the new catalog, you must manually rebuild the index before booting the server:
```bash
python build_index.py
```
This script will recalculate the 384-dimensional embeddings for all items and overwrite the `faiss.index` block.

### Running the Server

Execute the bootstrapper:
```bash
python run.py
```

The server will eagerly load the 384-dimensional vector database, initialize the ONNX engine, and attach to port 8000. 

## API Reference

The primary endpoint processes unstructured recipes and returns normalized catalog matching JSON arrays.

### HTTP POST Request

```bash
curl -X POST http://localhost:8000/api/shopping-list \
  -H "Content-Type: application/json" \
  -d '{
    "recipe_text": "I need 1 ripe avocado for a salad",
    "price_tier": "low"
  }'
```

### Response

```json
{
  "items": [
    {
      "ingredient_name": "ripe avocado",
      "article": {
        "Article_ID": "ART-0135",
        "Raw_Name": "Avocado Net (Unripe)",
        "Brand": "Picnic Basic",
        "Price": 2.5,
        "Quantity_Value": 700.0,
        "Quantity_Unit": "g",
        "Description": "Budget net of smaller, firm avocados. Needs ripening at home.",
        "Price_Per_Unit": 0.0035
      },
      "match_confidence": 0.6512,
      "packs_needed": 1,
      "total_quantity": 700.0,
      "total_quantity_unit": "g",
      "total_price": 2.5,
      "price_tier": "low",
      "is_optional": false
    }
  ],
  "total_cost": 2.5,
  "price_tier": "low",
  "recipe_text": "I need 1 ripe avocado for a salad",
  "parsed_ingredients_count": 1
}
```

Thank you again to Maya Budhdeo and the entire Picnic team for the inspiration.

<p align="center">
  <img src="https://media.tenor.com/BMSyfCkggfQAAAAM/picnic-picnicapp.gif" alt="Picnic Logo Animation" />
</p>
