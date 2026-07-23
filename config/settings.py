"""
Central Production Configuration
Egypt Education Real-Time RAG
"""

from __future__ import annotations


import os

from pathlib import Path

from dotenv import load_dotenv

from pydantic import BaseModel, Field



BASE_DIR = Path(__file__).resolve().parent.parent


load_dotenv(
    BASE_DIR / ".env"
)




# ==========================
# Secrets
# ==========================


def get_secret(
    key: str,
    default=""
):


    try:

        import streamlit as st


        value = st.secrets.get(
            key,
            None
        )


        if value:

            return str(value)


    except Exception:

        pass



    return os.getenv(
        key,
        default
    )



# HuggingFace Token
# Loaded from Streamlit Secrets or .env

HF_TOKEN = get_secret(
    "HF_TOKEN",
    ""
)




# ==========================
# Scraper
# ==========================


class ScraperConfig(BaseModel):


    timeout: int = 15


    max_pages: int = 8


    concurrent_requests: int = 10


    cache_ttl: int = 900


    delay: float = 0.1



    user_agent: str = (

        "Mozilla/5.0 "
        "Education-RAG-Bot"

    )



    ddg_max_results: int = 5



    sources: dict[str, list[str]] = {


        "curriculum": [

            "https://ar.wikipedia.org/wiki/التعليم_في_مصر",

            "https://ar.wikipedia.org/wiki/وزارة_التربية_والتعليم_(مصر)"

        ],


        "math": [

            "https://ar.wikipedia.org/wiki/رياضيات",

            "https://ar.wikipedia.org/wiki/الجبر"

        ],


        "science": [

            "https://ar.wikipedia.org/wiki/علوم",

            "https://ar.wikipedia.org/wiki/فيزياء",

            "https://ar.wikipedia.org/wiki/كيمياء"

        ],


        "arabic": [

            "https://ar.wikipedia.org/wiki/اللغة_العربية"

        ],


        "history": [

            "https://ar.wikipedia.org/wiki/تاريخ_مصر"

        ]

    }




# ==========================
# Retrieval
# ==========================


class RetrievalConfig(BaseModel):


    # Retrieval

    top_k_retrieve: int = 12


    top_k_rerank: int = 4



    # Chunking

    chunk_size: int = 750


    chunk_overlap: int = 120



    # Embedding
    # Lightweight multilingual model

    embedding_model: str = (

        "sentence-transformers/"
        "paraphrase-multilingual-MiniLM-L12-v2"

    )


    embedding_batch_size: int = 64



    # Hybrid

    bm25_weight: float = 0.35


    dense_weight: float = 0.65



    # Vector Store

    vector_dir: Path = (

        BASE_DIR /
        "data" /
        "vector_store"

    )


    faiss_file: str = "index.faiss"


    metadata_file: str = "metadata.pkl"





# ==========================
# Reranker
# ==========================


class RerankerConfig(BaseModel):


    enabled: bool = True


    # Lightweight Cross Encoder

    model: str = (

        "cross-encoder/"
        "ms-marco-MiniLM-L-6-v2"

    )


    top_k: int = 4





# ==========================
# LLM
# ==========================


class LLMConfig(BaseModel):


    openrouter_api_key: str = Field(

        default_factory=lambda:

        get_secret(
            "OPENROUTER_API_KEY",
            ""
        )

    )



    model: str = Field(

        default_factory=lambda:

        get_secret(
            "LLM_MODEL",
            "google/gemini-2.0-flash-exp:free"
        )

    )



    max_tokens: int = 1200


    temperature: float = 0.15


    timeout: int = 90


    streaming: bool = True





# ==========================
# Cache
# ==========================


class CacheConfig(BaseModel):


    enabled: bool = True


    query_cache_size: int = 500


    document_cache_size: int = 1000


    ttl_seconds: int = 900





# ==========================
# Application
# ==========================


class AppConfig(BaseModel):


    title: str = (

        "نظام الذكاء الاصطناعي "
        "للتعليم المصري"

    )


    environment: str = "production"


    log_level: str = "INFO"



    data_dir: Path = (

        BASE_DIR /
        "data"

    )


    raw_dir: Path = (

        BASE_DIR /
        "data" /
        "raw"

    )


    processed_dir: Path = (

        BASE_DIR /
        "data" /
        "processed"

    )





# ==========================
# Instances
# ==========================


scraper_cfg = ScraperConfig()


retrieval_cfg = RetrievalConfig()


reranker_cfg = RerankerConfig()


cache_cfg = CacheConfig()


llm_cfg = LLMConfig()


app_cfg = AppConfig()





# ==========================
# Create folders
# ==========================


for path in [

    app_cfg.raw_dir,

    app_cfg.processed_dir,

    retrieval_cfg.vector_dir

]:

    path.mkdir(

        parents=True,

        exist_ok=True

    )
