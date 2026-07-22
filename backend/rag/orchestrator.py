"""
Real-Time RAG Orchestrator
Async pipeline controller
"""


from __future__ import annotations


import asyncio

from typing import Generator


from loguru import logger


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
    RetrievedChunk,
    RetrieverType
)


from backend.rag.chain import (
    generate_answer_async,
    stream_answer_async,
    RAGAnswer
)


from config.settings import retrieval_cfg





# ==========================
# Simple memory cache
# ==========================


PIPELINE_CACHE={}





def cache_key(question):

    return question.strip().lower()





# ==========================
# Status
# ==========================


def update_status(
    callback,
    msg
):

    logger.info(msg)

    if callback:

        callback(msg)





# ==========================
# Async Pipeline
# ==========================


async def run_rag_pipeline_async(

    question:str,

    retriever_strategy:RetrieverType="hybrid",

    stream=True,

    status_callback=None

):


    update_status(
        status_callback,
        "Analyzing query"
    )



    meta:QueryMetadata = analyze_query(
        question
    )



    key=cache_key(
        meta.search_query
    )



    # ======================
    # Check Cache
    # ======================


    if key in PIPELINE_CACHE:


        chunks=PIPELINE_CACHE[key]


        logger.info(
            "Using cached chunks"
        )


    else:



        update_status(
            status_callback,
            "Live scraping"
        )


        docs=await async_scrape_for_query(
            meta
        )



        if not docs:


            return RAGAnswer(

                answer=
                "لم يتم العثور على مصادر.",

                sources=[],

                retriever_used="none",

                chunks_retrieved=0

            )



        update_status(

            status_callback,

            "Processing documents"

        )



        chunks=process_documents(
            docs
        )



        PIPELINE_CACHE[key]=chunks




    update_status(

        status_callback,

        "Building retrieval"

    )



    retriever=build_retriever(
        retriever_strategy
    )



    retriever.index(
        chunks
    )



    update_status(

        status_callback,

        "Searching"

    )



    retrieved=retriever.search(

        question,

        retrieval_cfg.top_k_rerank

    )



    update_status(

        status_callback,

        "Generating answer"

    )



    if stream:


        async for token in stream_answer_async(

            question,

            retrieved

        ):

            yield token


    else:


        return await generate_answer_async(

            question,

            retrieved

        )





# ==========================
# Streamlit wrapper
# ==========================


def run_rag_pipeline(

    question,

    retriever_strategy="hybrid",

    stream=True,

    status_callback=None

):


    return asyncio.run(

        run_rag_pipeline_async(

            question,

            retriever_strategy,

            stream,

            status_callback

        )

    )





# ==========================
# Metadata API
# ==========================


def get_pipeline_metadata(
    question
):


    meta=analyze_query(
        question
    )


    return {


        "grade":
        meta.grade,


        "subject":
        meta.subject,


        "term":
        meta.term,


        "domain":
        meta.domain,


        "intent":
        meta.intent,


        "live":
        meta.needs_live_search,


        "query":
        meta.search_query

    }
