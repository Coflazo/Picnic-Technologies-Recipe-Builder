# spins up the fastapi app and serves the frontend

from __future__ import annotations

import logging
import json
import faiss
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .models import Article, ParsedIngredient, PriceTier, ShoppingList, load_catalog
from .pipeline import RecipeBuilderPipeline

logger = logging.getLogger(__name__)


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise RuntimeError(
            f"Missing required {description}: {path}. "
            "Please follow the README setup steps to provide local model/index assets."
        )

# schemas for request/response
class ParseRequest(BaseModel):
    recipe_text: str = Field(..., min_length=10)

class ShoppingListRequest(BaseModel):
    recipe_text: str = Field(..., min_length=10)
    price_tier: PriceTier = Field(default=PriceTier.MEDIUM)

class HealthResponse(BaseModel):
    status: str = "ok"
    articles_loaded: int = 0
    index_ready: bool = False

GLOBAL_STATE = {}

# load up the models before we start taking requests
@asynccontextmanager
async def lifespan(app: FastAPI):
    from rich.console import Console
    console = Console()
    console.print("\n[bold #198917]Application Bootstrapping[/bold #198917]")
    
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    articles = load_catalog()
    GLOBAL_STATE["articles"] = articles
    GLOBAL_STATE["article_map"] = {a.article_id: a for a in articles}
    console.print(f"  [#198917]OK[/#198917] [#eeeeee]Normalized Catalog loaded ({len(articles)} items)[/#eeeeee]")
    
    faiss_path = _PROJECT_ROOT / "data" / "index" / "faiss.index"
    article_ids_path = _PROJECT_ROOT / "data" / "index" / "article_ids.json"
    _require_file(faiss_path, "FAISS index")
    _require_file(article_ids_path, "article id map")
    GLOBAL_STATE["faiss_index"] = faiss.read_index(str(faiss_path))
    
    with open(article_ids_path, "r") as f:
        GLOBAL_STATE["article_ids"] = json.load(f)
    console.print("  [#198917]OK[/#198917] [#eeeeee]FAISS Vector Engine mapped[/#eeeeee]")
        
    from transformers import AutoTokenizer
    import onnxruntime as ort
    
    onnx_model_dir = _PROJECT_ROOT / "onnx_model"
    onnx_model_path = onnx_model_dir / "model.onnx"
    _require_file(onnx_model_path, "ONNX model file")
    GLOBAL_STATE["tokenizer"] = AutoTokenizer.from_pretrained(str(onnx_model_dir), local_files_only=True)
    GLOBAL_STATE["onnx_model"] = ort.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
    console.print("  [#198917]OK[/#198917] [#eeeeee]ONNX Embedding Model & Pipeline ready[/#eeeeee]")
    
    from gliner import GLiNER
    # Forcing CPU map_location to bypass Apple Silicon MPS memory swap crashes
    GLOBAL_STATE["gliner"] = GLiNER.from_pretrained("urchade/gliner_small-v2.1", map_location="cpu")
    console.print("  [#198917]OK[/#198917] [#eeeeee]GLiNER NLP Model initialized[/#eeeeee]")
    
    console.print("\n[bold #198917]System Ready. Waiting for traffic...[/bold #198917]\n")
    
    yield
    console.print("[bold #ea4b4b]Shutting down server...[/bold #ea4b4b]")
    GLOBAL_STATE.clear()

app = FastAPI(lifespan=lifespan)

# let the frontend talk to us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/parse", response_model=list[ParsedIngredient])
async def parse_recipe_endpoint(req: ParseRequest):
    try:
        from .pipeline import RecipeBuilderPipeline
        pipeline = RecipeBuilderPipeline(GLOBAL_STATE)
        return pipeline.parse(req.recipe_text)
    except Exception as e:
        logger.exception("failed to parse recipe")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/shopping-list", response_model=ShoppingList)
async def generate_shopping_list(req: ShoppingListRequest):
    try:
        from .pipeline import RecipeBuilderPipeline
        pipeline = RecipeBuilderPipeline(GLOBAL_STATE)
        return pipeline.generate_shopping_list(
            recipe_text=req.recipe_text,
            price_tier=req.price_tier,
        )
    except Exception as e:
        logger.exception("failed to generate list")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/articles", response_model=list[Article])
async def list_articles():
    return load_catalog()

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="ok",
        articles_loaded=len(GLOBAL_STATE.get("articles", [])),
        index_ready=bool(GLOBAL_STATE.get("faiss_index")),
    )

_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

# serve the single page app if it exists
if _FRONTEND_DIR.exists():
    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(_FRONTEND_DIR / "index.html"))

    @app.get("/{path:path}")
    async def serve_static(path: str):
        if path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        file_path = _FRONTEND_DIR / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return FileResponse(str(_FRONTEND_DIR / "index.html"))
