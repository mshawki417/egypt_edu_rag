"""
backend/rag/orchestrator.py
Top-level pipeline — wires all components together.
Pipeline runs ONCE per question (no double-run).
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


def _stream_msg(msg: str):
    """Helper: return a generator that yields a single message."""
    def _gen():
        yield msg
    return _gen()


def run_rag_pipeline(
    question: str,
    retriever_strategy: RetrieverType = "hybrid",
    stream: bool = True,
    status_callback=None,   # optional: callable(step: str) for live UI updates
) -> Generator[str, None, None] | RAGAnswer:
    """
    Full RAG pipeline. Steps:
      1. Analyze query
      2. Live scrape
      3. Preprocess + chunk
      4. Build index
      5. Retrieve top-k
      6. Generate answer (streaming or batch)

    status_callback(step) is called at each step so the UI can show progress.
    """
    def _status(msg: str):
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    _status(f"=== Pipeline START | strategy={retriever_strategy} | stream={stream} ===")

    # 1. Query analysis
    _status("step:analyze")
    meta: QueryMetadata = analyze_query(question)

    # 2. Live scraping
    _status("step:scrape")
    raw_docs = scrape_for_query(meta)
    if not raw_docs:
        logger.warning("No documents scraped")
        _status("step:error_scrape")
        if stream:
            return _stream_msg("لم أتمكن من الوصول إلى المصادر. يرجى التحقق من اتصالك بالإنترنت.")
        return generate_answer(question, [])

    # 3. Preprocessing
    _status("step:chunk")
    chunks = process_documents(raw_docs)
    logger.info(f"Total chunks after processing: {len(chunks)}")

    if not chunks:
        _status("step:error_chunk")
        if stream:
            return _stream_msg("لم يتم استخراج محتوى كافٍ من المصادر.")
        return generate_answer(question, [])

    # 4. Index
    _status("step:index")
    retriever = build_retriever(retriever_strategy)
    retriever.index(chunks)

    # 5. Retrieve
    _status("step:retrieve")
    top_k = retrieval_cfg.top_k_rerank
    retrieved: list[RetrievedChunk] = retriever.search(question, top_k)
    logger.info(f"Retrieved {len(retrieved)} chunks")

    # 6. Generate
    _status("step:generate")
    if stream:
        return stream_answer(question, retrieved)
    return generate_answer(question, retrieved)


def get_pipeline_metadata(question: str) -> dict:
    meta = analyze_query(question)
    return {
        "grade": meta.grade,
        "subject": meta.subject,
        "term": meta.term,
        "domain": meta.domain,
        "needs_live_search": meta.needs_live_search,
        "search_query": meta.search_query,
    }
