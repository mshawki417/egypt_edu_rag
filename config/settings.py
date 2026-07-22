"""
config/settings.py
Central configuration — reads Streamlit Cloud secrets first, then .env fallback.
"""
from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit secrets first, then env vars, then default."""
    try:
        import streamlit as st
        val = st.secrets.get(key, None)
        if val:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


class ScraperConfig(BaseModel):
    timeout: int   = 20
    max_pages: int = 5
    delay: float   = 0.5
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    sources: dict[str, list[str]] = {
        "curriculum": [
            "https://ar.wikipedia.org/wiki/التعليم_في_مصر",
            "https://ar.wikipedia.org/wiki/وزارة_التربية_والتعليم_(مصر)",
        ],
        "math": [
            "https://ar.wikipedia.org/wiki/رياضيات",
            "https://ar.wikipedia.org/wiki/الجبر",
        ],
        "science": [
            "https://ar.wikipedia.org/wiki/علوم",
            "https://ar.wikipedia.org/wiki/فيزياء",
            "https://ar.wikipedia.org/wiki/كيمياء",
            "https://ar.wikipedia.org/wiki/أحياء",
        ],
        "arabic": [
            "https://ar.wikipedia.org/wiki/اللغة_العربية",
            "https://ar.wikipedia.org/wiki/النحو_العربي",
        ],
        "history": [
            "https://ar.wikipedia.org/wiki/تاريخ_مصر",
            "https://ar.wikipedia.org/wiki/تاريخ_مصر_الحديث",
        ],
        "exams": [
            "https://ar.wikipedia.org/wiki/الثانوية_العامة_في_مصر",
        ],
        "news": [
            "https://ar.wikipedia.org/wiki/التعليم_في_مصر",
        ],
        "schools": [
            "https://ar.wikipedia.org/wiki/التعليم_الابتدائي_في_مصر",
            "https://ar.wikipedia.org/wiki/التعليم_الثانوي_في_مصر",
        ],
    }
    ddg_max_results: int = 5


class RetrievalConfig(BaseModel):
    top_k_retrieve: int  = 8
    top_k_rerank: int    = 4
    chunk_size: int      = 800
    chunk_overlap: int   = 80
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    bm25_weight: float   = 0.4


class LLMConfig(BaseModel):
    openrouter_api_key: str = ""
    model: str = "mistralai/mistral-7b-instruct:free"
    max_tokens: int    = 1024
    temperature: float = 0.1

    def __init__(self, **data):
        super().__init__(**data)
        # Read secrets at runtime, not at import time
        self.openrouter_api_key = _get_secret("OPENROUTER_API_KEY", "")
        self.model = _get_secret("LLM_MODEL", "mistralai/mistral-7b-instruct:free")


class AppConfig(BaseModel):
    title: str     = "نظام الذكاء الاصطناعي للتعليم المصري"
    log_level: str = "INFO"
    data_dir: Path       = BASE_DIR / "data"
    raw_dir: Path        = BASE_DIR / "data" / "raw"
    processed_dir: Path  = BASE_DIR / "data" / "processed"


scraper_cfg   = ScraperConfig()
retrieval_cfg = RetrievalConfig()
llm_cfg       = LLMConfig()
app_cfg       = AppConfig()

app_cfg.raw_dir.mkdir(parents=True, exist_ok=True)
app_cfg.processed_dir.mkdir(parents=True, exist_ok=True)
