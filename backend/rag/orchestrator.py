"""
Real-Time RAG Orchestrator
Production RAG pipeline controller
"""

from __future__ import annotations

import asyncio

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
    RetrieverType
)


from backend.rag.chain import (
    generate_answer_async,
    RAGAnswer
)


from config.settings import retrieval_cfg



PIPELINE_CACHE = {}



def cache_key(question):

    return question.strip().lower()



def update_status(callback,msg):

    logger.info(msg)

    if callback:

        callback(msg)





async def run_rag_pipeline_async(

    question:str,

    retriever_strategy:RetrieverType="hybrid",

    status_callback=None

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

        retrieval_cfg.top_k_rerank

    )



    update_status(
        status_callback,
        "Generating answer"
    )



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

            status_callback

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
