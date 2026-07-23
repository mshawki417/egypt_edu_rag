"""
Real-Time RAG Orchestrator
Production RAG pipeline controller
"""

from __future__ import annotations


import asyncio

from threading import Lock

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


from config.settings import (
    retrieval_cfg,
    reranker_cfg
)





# =====================================
# Cache
# =====================================


PIPELINE_CACHE = TTLCache(

    maxsize=100,

    ttl=300

)


CACHE_LOCK = Lock()



# =====================================
# Models
# =====================================


_reranker_model = None



_RETRIEVER_CACHE = {}





# =====================================
# Cache Utils
# =====================================


def cache_key(question: str):

    return (

        question

        .strip()

        .lower()

    )





def clear_pipeline_cache():

    with CACHE_LOCK:

        PIPELINE_CACHE.clear()


    logger.info(

        "Pipeline cache cleared"

    )





# =====================================
# Retriever Manager
# =====================================


def get_retriever(

    strategy: RetrieverType

):


    if strategy not in _RETRIEVER_CACHE:


        logger.info(

            f"Creating retriever: {strategy}"

        )


        _RETRIEVER_CACHE[strategy] = build_retriever(

            strategy

        )


    return _RETRIEVER_CACHE[strategy]





# =====================================
# Reranker
# =====================================


def get_reranker():


    global _reranker_model



    if _reranker_model is None:


        logger.info(

            "Loading reranker model..."

        )


        from sentence_transformers import CrossEncoder



        _reranker_model = CrossEncoder(

            reranker_cfg.model,

            max_length=512

        )



        logger.info(

            "Reranker loaded"

        )



    return _reranker_model






# =====================================
# Utils
# =====================================


def update_status(

    callback,

    message

):


    logger.info(message)


    if callback:

        callback(message)






# =====================================
# Main Async Pipeline
# =====================================


async def run_rag_pipeline_async(


    question: str,


    retriever_strategy: RetrieverType = "hybrid",


    status_callback=None,


    stream=False


):


    update_status(

        status_callback,

        "Analyzing query"

    )



    meta: QueryMetadata = analyze_query(

        question

    )



    key = cache_key(

        meta.search_query

    )



    chunks = None



    # -----------------------------
    # Get cached chunks
    # -----------------------------


    with CACHE_LOCK:

        cached = PIPELINE_CACHE.get(key)



    if cached:


        logger.info(

            "Using cached documents"

        )


        chunks = cached



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



        with CACHE_LOCK:


            PIPELINE_CACHE[key] = chunks





    # -----------------------------
    # Retrieval
    # -----------------------------


    update_status(

        status_callback,

        "Building retrieval"

    )



    retriever = get_retriever(

        retriever_strategy

    )



    retriever.index(

        chunks

    )



    update_status(

        status_callback,

        "Searching"

    )



    retrieval_limit = min(

        retrieval_cfg.top_k_retrieve,

        15

    )



    retrieved = retriever.search(

        question,

        retrieval_limit

    )






    # -----------------------------
    # Reranking
    # -----------------------------


    if reranker_cfg.enabled and retrieved:



        update_status(

            status_callback,

            "Reranking documents"

        )


        try:


            reranker = get_reranker()



            pairs = [

                [

                    question,

                    item.chunk.text

                ]

                for item in retrieved

            ]



            scores = reranker.predict(

                pairs,

                batch_size=8,

                show_progress_bar=False

            )



            for item, score in zip(

                retrieved,

                scores

            ):


                item.score = float(score)



            retrieved.sort(

                key=lambda x: x.score,

                reverse=True

            )



            retrieved = retrieved[

                :reranker_cfg.top_k

            ]



            logger.info(

                f"Reranking completed: {len(retrieved)} chunks"

            )



        except Exception as e:


            logger.exception(

                f"Reranking failed: {e}"

            )



            retrieved = retrieved[

                :reranker_cfg.top_k

            ]



    else:


        retrieved = retrieved[

            :reranker_cfg.top_k

        ]






    # -----------------------------
    # Generation
    # -----------------------------


    update_status(

        status_callback,

        "Generating answer"

    )




    if stream:


        return stream_answer_async(

            question,

            retrieved

        )




    return await generate_answer_async(

        question,

        retrieved

    )








# =====================================
# Sync Wrapper
# =====================================


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







# =====================================
# Metadata
# =====================================


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
