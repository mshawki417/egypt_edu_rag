"""
config/settings.py
Central configuration — loaded once, used everywhere.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class ScraperConfig(BaseModel):
    timeout: int = int(os.getenv("SCRAPER_TIMEOUT", 30))
    max_pages: int = int(os.getenv("MAX_PAGES_PER_SOURCE", 5))
    delay: float = float(os.getenv("REQUEST_DELAY", 1.5))
    user_agent: str = (
        "Mozilla/5.0 (compatible; EduRAGBot/1.0; +https://github.com/your/repo)"
    )
    sources: dict[str, list[str]] = {
        "curriculum": [
            "https://www.moe.gov.eg",
            "https://studentbooks.moe.gov.eg",
        ],
        "schools":  ["https://emis.gov.eg"],
        "exams":    ["https://www.moe.gov.eg/ar/examinations"],
        "news":     ["https://www.moe.gov.eg/ar/news"],
    }


class RetrievalConfig(BaseModel):
    top_k_retrieve: int = int(os.getenv("TOP_K_RETRIEVE", 10))
    top_k_rerank: int  = int(os.getenv("TOP_K_RERANK", 4))
    chunk_size: int    = int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", 64))
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    bm25_weight: float = 0.4


class LLMConfig(BaseModel):
    # ── OpenRouter ────────────────────────────────────────────────────────────
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # Free models on OpenRouter (change in .env to switch):
    #   meta-llama/llama-3.1-8b-instruct:free   ← fast, good Arabic
    #   mistralai/mistral-7b-instruct:free
    #   google/gemma-2-9b-it:free
    #   qwen/qwen-2-7b-instruct:free             ← excellent multilingual
    model: str = os.getenv(
        "LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
    )
    max_tokens: int   = 2048
    temperature: float = 0.1


class AppConfig(BaseModel):
    title: str     = os.getenv("APP_TITLE", "نظام الذكاء الاصطناعي للتعليم المصري")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    data_dir: Path       = BASE_DIR / "data"
    raw_dir: Path        = BASE_DIR / "data" / "raw"
    processed_dir: Path  = BASE_DIR / "data" / "processed"


scraper_cfg   = ScraperConfig()
retrieval_cfg = RetrievalConfig()
llm_cfg       = LLMConfig()
app_cfg       = AppConfig()

app_cfg.raw_dir.mkdir(parents=True, exist_ok=True)
app_cfg.processed_dir.mkdir(parents=True, exist_ok=True)
