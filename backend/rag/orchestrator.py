"""
backend/rag/orchestrator.py

Top-level pipeline that wires all components together.
Called directly from the Streamlit frontend.
"""
from __future__ import annotations
from typing import Generator, Literal

from loguru import logger

from backend.scraper.query_analyzer import analyze_query, QueryMetadata
from backend.scraper.live_scraper import scrape_for_query
from backend.preprocessing.chunker import process_documents
from backend.retrieval.retriever import build_retriever, RetrievedChunk, RetrieverType
from backend.rag.chain import generate_answer, stream_answer, RAGAnswer
from config.settings import retrieval_cfg


def run_rag_pipeline(
    question: str,
    retriever_strategy: RetrieverType = "hybrid",
    stream: bool = True,
) -> Generator[str, None, None] | RAGAnswer:
    """
    Full end-to-end RAG pipeline.

    Steps:
      1. Analyze query → extract metadata
      2. Live scrape MOE sources
      3. Preprocess (clean + chunk)
      4. Build ephemeral index
      5. Retrieve top-k chunks
      6. Generate LLM answer (streaming or batch)

    Args:
        question: The user's Arabic question.
        retriever_strategy: "bm25" | "dense" | "hybrid"
        stream: If True, returns a generator of text tokens.

    Returns:
        RAGAnswer (batch) or Generator[str] (streaming).
    """
    logger.info(f"=== Pipeline START | strategy={retriever_strategy} | stream={stream} ===")

    # ── 1. Query analysis ──────────────────────────────────────────────────────
    meta: QueryMetadata = analyze_query(question)

    # ── 2. Live scraping ───────────────────────────────────────────────────────
    raw_docs = scrape_for_query(meta)
    if not raw_docs:
        logger.warning("No documents scraped — returning empty answer")
        if stream:
            def _empty():
                yield "لم أتمكن من الوصول إلى المصادر. يرجى التحقق من اتصالك."
            return _empty()
        return generate_answer(question, [])

    # ── 3. Preprocessing ───────────────────────────────────────────────────────
    chunks = process_documents(raw_docs)
    logger.info(f"Total chunks after processing: {len(chunks)}")

    if not chunks:
        if stream:
            def _empty():
                yield "لم يتم استخراج محتوى كافٍ من المصادر."
            return _empty()
        return generate_answer(question, [])

    # ── 4. Index ───────────────────────────────────────────────────────────────
    retriever = build_retriever(retriever_strategy)
    retriever.index(chunks)

    # ── 5. Retrieve ────────────────────────────────────────────────────────────
    top_k = retrieval_cfg.top_k_rerank
    retrieved: list[RetrievedChunk] = retriever.search(question, top_k)
    logger.info(f"Retrieved {len(retrieved)} chunks")

    # ── 6. Generate ────────────────────────────────────────────────────────────
    if stream:
        return stream_answer(question, retrieved)
    return generate_answer(question, retrieved)


def get_pipeline_metadata(question: str) -> dict:
    """Return debug info about how a question would be routed (no LLM call)."""
    meta = analyze_query(question)
    return {
        "grade": meta.grade,
        "subject": meta.subject,
        "term": meta.term,
        "domain": meta.domain,
        "needs_live_search": meta.needs_live_search,
        "search_query": meta.search_query,
    }
