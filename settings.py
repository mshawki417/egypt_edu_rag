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
    delay: float = float(os.getenv("REQUEST_DELAY", 1.0))
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )

    # ✅ مصادر جديدة كلها تقبل الوصول من Streamlit Cloud
    sources: dict[str, list[str]] = {
        "curriculum": [
            # ويكيبيديا العربية - مفتوحة دايماً ✅
            "https://ar.wikipedia.org/wiki/التعليم_في_مصر",
            "https://ar.wikipedia.org/wiki/وزارة_التربية_والتعليم_(مصر)",
            "https://ar.wikipedia.org/wiki/المنهج_الدراسي_المصري",
            # أرشيف الكتب المدرسية المصرية على Archive.org ✅
            "https://archive.org/search?query=كتب+مدرسية+مصرية&output=json",
            # بوابة تعليمية عربية مفتوحة ✅
            "https://www.almaany.com",
        ],
        "schools": [
            "https://ar.wikipedia.org/wiki/التعليم_الابتدائي_في_مصر",
            "https://ar.wikipedia.org/wiki/التعليم_الثانوي_في_مصر",
        ],
        "exams": [
            "https://ar.wikipedia.org/wiki/الثانوية_العامة_في_مصر",
            "https://ar.wikipedia.org/wiki/امتحانات_الثانوية_العامة_المصرية",
        ],
        "news": [
            # أخبار التعليم من مصادر عربية مفتوحة ✅
            "https://ar.wikipedia.org/wiki/التعليم_في_مصر",
        ],
        "math": [
            "https://ar.wikipedia.org/wiki/رياضيات",
            "https://ar.wikipedia.org/wiki/الجبر",
            "https://ar.wikipedia.org/wiki/الهندسة_الإقليدية",
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
    }

    # ✅ مصادر DuckDuckGo للبحث الحي - بديل Google
    ddg_regions: list[str] = ["eg-ar", "wt-ar"]
    ddg_max_results: int = 8


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
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
    model: str = os.getenv(
        "LLM_MODEL", "meta-llama/llama-3.1-8b-instruct:free"
    )
    max_tokens: int    = 2048
    temperature: float = 0.1


class AppConfig(BaseModel):
    title: str     = os.getenv("APP_TITLE", "نظام الذكاء الاصطناعي للتعليم المصري")
    log_level: str = os.getenv("LOG_LEVEL", "DEBUG")   # ✅ DEBUG عشان نشوف التفاصيل
    data_dir: Path       = BASE_DIR / "data"
    raw_dir: Path        = BASE_DIR / "data" / "raw"
    processed_dir: Path  = BASE_DIR / "data" / "processed"


scraper_cfg   = ScraperConfig()
retrieval_cfg = RetrievalConfig()
llm_cfg       = LLMConfig()
app_cfg       = AppConfig()

app_cfg.raw_dir.mkdir(parents=True, exist_ok=True)
app_cfg.processed_dir.mkdir(parents=True, exist_ok=True)
