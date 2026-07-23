"""
Real-Time RAG Orchestrator
Production RAG pipeline controller
"""

from __future__ import annotations

import asyncio

from loguru import logger
from cachetools import TTLCache


from backend.scraper.query_analyzer import (
    analyze_query,
    QueryMetadata
)


from backend.scraper.live_scraper import (
    async_scrape_for_query
)


from backend.preprocessing.chunker import (
    process_documents
)


from backend.retrieval.retriever import (
    build_retriever,
    RetrieverType
)


from backend.rag.chain import (
    generate_answer_async,
    stream_answer_async,
    RAGAnswer
)


from config.settings import retrieval_cfg, reranker_cfg



PIPELINE_CACHE = TTLCache(maxsize=1000, ttl=3600)

_reranker_model = None

def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(reranker_cfg.model)
    return _reranker_model



def cache_key(question):

    return question.strip().lower()



def update_status(callback,msg):

    logger.info(msg)

    if callback:

        callback(msg)





async def run_rag_pipeline_async(

    question:str,

    retriever_strategy:RetrieverType="hybrid",

    status_callback=None,

    stream=False

):


    update_status(
        status_callback,
        "Analyzing query"
    )


    meta:QueryMetadata = analyze_query(
        question
    )


    key = cache_key(
        meta.search_query
    )



    if key in PIPELINE_CACHE:


        chunks = PIPELINE_CACHE[key]


    else:


        update_status(
            status_callback,
            "Live scraping"
        )


        docs = await async_scrape_for_query(
            meta
        )


        if not docs:


            return RAGAnswer(

                answer="لم يتم العثور على مصادر.",

                sources=[],

                retriever_used="none",

                chunks_retrieved=0

            )


        update_status(
            status_callback,
            "Processing documents"
        )


        chunks = process_documents(
            docs
        )


        PIPELINE_CACHE[key]=chunks




    update_status(
        status_callback,
        "Building retrieval"
    )


    retriever = build_retriever(
        retriever_strategy
    )


    retriever.index(
        chunks
    )



    update_status(
        status_callback,
        "Searching"
    )


    retrieved = retriever.search(

        question,

        retrieval_cfg.top_k_retrieve

    )


    if reranker_cfg.enabled and retrieved:
        update_status(status_callback, "Reranking documents")
        reranker = get_reranker()
        pairs = [[question, item.chunk.text] for item in retrieved]
        scores = reranker.predict(pairs)
        
        for item, score in zip(retrieved, scores):
            item.score = float(score)
            
        retrieved.sort(key=lambda x: x.score, reverse=True)
        retrieved = retrieved[:reranker_cfg.top_k]
    else:
        retrieved = retrieved[:reranker_cfg.top_k]



    update_status(
        status_callback,
        "Generating answer"
    )



    if stream:
        return stream_answer_async(question, retrieved)

    return await generate_answer_async(

        question,

        retrieved

    )






def run_rag_pipeline(

    question,

    retriever_strategy="hybrid",

    stream=False,

    status_callback=None

):


    return asyncio.run(

        run_rag_pipeline_async(

            question,

            retriever_strategy,

            status_callback,

            stream=stream

        )

    )





def get_pipeline_metadata(question):


    meta = analyze_query(
        question
    )


    return {

        "grade":

        meta.grade,


        "subject":

        meta.subject,


        "intent":

        meta.intent,


        "live":

        meta.needs_live_search,


        "query":

        meta.search_query

    }
