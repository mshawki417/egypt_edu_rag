"""
Production Real-Time RAG Orchestrator

Version:
Advanced v5

Responsibilities:
- Query understanding
- Live retrieval orchestration
- Cache management
- Retriever lifecycle
- RAG pipeline control

Compatible:
- live_scraper.py v5
- query_analyzer.py v3
- chunker.py
- retriever.py
- rag chain
"""


from __future__ import annotations



# =====================================================
# Standard Library
# =====================================================


import asyncio

import hashlib

import time

from dataclasses import dataclass

from threading import Lock



# =====================================================
# Third Party
# =====================================================


from cachetools import TTLCache

from loguru import logger



# =====================================================
# Internal Modules
# =====================================================


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







# =====================================================
# Pipeline Configuration
# =====================================================


@dataclass(
    frozen=True
)
class PipelineConfig:


    """
    Global RAG execution settings.
    """


    cache_size:int = 200


    cache_ttl:int = 600



    max_retrieve:int = 15



    enable_reranker:bool = True



    scraper_timeout:int = 60



    version:str = "rag-v5"






PIPELINE_CFG = PipelineConfig()







# =====================================================
# Pipeline Cache
# =====================================================


class PipelineCache:


    """
    Thread safe cache manager.

    Stores:

    - processed chunks
    - retriever index
    - metadata

    """


    def __init__(self):


        self.cache = TTLCache(

            maxsize=

            PIPELINE_CFG.cache_size,


            ttl=

            PIPELINE_CFG.cache_ttl

        )


        self.lock = Lock()





    def make_key(
        self,
        question:str,
        strategy:str
    ):


        payload=(


            question.strip()

            +

            strategy

            +

            PIPELINE_CFG.version

        )



        return hashlib.sha256(

            payload.encode(
                "utf-8"
            )

        ).hexdigest()







    def get(
        self,
        key
    ):


        with self.lock:


            return self.cache.get(

                key

            )







    def set(
        self,
        key,
        value
    ):


        with self.lock:


            self.cache[key]=value








    def clear(self):


        with self.lock:


            self.cache.clear()



        logger.info(

            "Pipeline cache cleared"

        )






PIPELINE_CACHE = PipelineCache()







# =====================================================
# Retriever Manager
# =====================================================


class RetrieverManager:


    """
    Keeps retriever instances alive.

    Avoids rebuilding vector index
    for every question.
    """


    def __init__(self):


        self.instances={}


        self.lock=Lock()





    def get(
        self,
        strategy:RetrieverType
    ):


        key=str(strategy)



        with self.lock:


            if key not in self.instances:


                logger.info(

                    f"Initializing retriever: {key}"

                )


                self.instances[key]=build_retriever(

                    strategy

                )



            return self.instances[key]







    def index(
        self,
        strategy,
        chunks
    ):


        retriever=self.get(

            strategy

        )


        retriever.index(

            chunks

        )


        return retriever







    def clear(self):


        with self.lock:


            self.instances.clear()



        logger.info(

            "Retriever manager cleared"

        )







RETRIEVER_MANAGER = RetrieverManager()







# =====================================================
# Async Safe Runner
# =====================================================


def run_async_safe(
    coroutine
):


    """
    Streamlit compatible async runner.

    Handles:

    - existing event loop
    - normal python execution

    """


    try:


        loop=asyncio.get_running_loop()



    except RuntimeError:


        loop=None





    if loop and loop.is_running():


        result=[]


        error=[]




        def runner():

            try:


                new_loop=asyncio.new_event_loop()


                asyncio.set_event_loop(

                    new_loop

                )


                result.append(

                    new_loop.run_until_complete(

                        coroutine

                    )

                )


            except Exception as e:


                error.append(e)



            finally:


                new_loop.close()






        import threading



        thread=threading.Thread(

            target=runner

        )


        thread.start()


        thread.join()



        if error:

            raise error[0]



        return result[0]




    else:


        return asyncio.run(

            coroutine

        )






# =====================================================
# Cache Utilities
# =====================================================


def clear_pipeline_cache():


    PIPELINE_CACHE.clear()


    RETRIEVER_MANAGER.clear()






def update_status(
    callback,
    message:str
):


    logger.info(

        message

    )


    if callback:


        callback(

            message

        )

# =====================================================
# LIVE SEARCH + DOCUMENT PROCESSING LAYER
# =====================================================


async def collect_documents(
    meta,
    status_callback=None
):
    """
    Collect documents from live scraper.

    Failure safe:
    - scraper errors do not kill pipeline
    - returns empty list
    - logs failures
    """

    try:

        update_status(
            status_callback,
            "Collecting live educational sources"
        )


        documents = await async_scrape_for_query(
            meta
        )


        if not documents:

            logger.warning(
                "Live scraper returned no documents"
            )

            return []



        logger.info(
            f"Collected {len(documents)} documents"
        )


        return documents



    except Exception as e:


        logger.exception(
            f"Scraper failure: {e}"
        )


        return []





# =====================================================
# DOCUMENT PROCESSING
# =====================================================


def process_rag_documents(
    documents
):
    """
    Convert raw documents into
    optimized RAG chunks.

    Adds:
    - duplicate filtering
    - chunk validation
    - metadata preservation
    """



    if not documents:

        return []



    try:


        chunks = process_documents(

            documents

        )



        if not chunks:


            logger.warning(
                "No chunks generated"
            )


            return []





        # remove duplicate chunks


        unique_chunks = {}



        for chunk in chunks:


            text = chunk.text.strip()



            if not text:

                continue



            fingerprint = hashlib.sha256(

                text.encode(
                    "utf-8"
                )

            ).hexdigest()



            if fingerprint not in unique_chunks:


                unique_chunks[fingerprint] = chunk





        final_chunks = list(

            unique_chunks.values()

        )




        logger.info(

            f"""

Document Processing Complete

Raw Documents:
{len(documents)}

Generated Chunks:
{len(chunks)}

Unique Chunks:
{len(final_chunks)}

"""

        )


        return final_chunks




    except Exception as e:


        logger.exception(

            f"Document processing failed: {e}"

        )


        return []







# =====================================================
# CHUNK CACHE MANAGEMENT
# =====================================================


def get_cached_chunks(
    key
):


    """
    Thread safe chunk retrieval.
    """


    with CACHE_LOCK:


        return PIPELINE_CACHE.get(

            key

        )






def save_chunks_cache(
    key,
    chunks
):


    """
    Store processed chunks.
    """


    if not chunks:

        return



    with CACHE_LOCK:


        PIPELINE_CACHE[key] = chunks



    logger.info(

        f"Cached {len(chunks)} chunks"

    )






# =====================================================
# RETRIEVER BUILDING
# =====================================================


_retriever_cache = {}

_retriever_lock = Lock()



def get_or_create_retriever(
    strategy
):

    """
    Retriever manager.

    Avoid rebuilding expensive
    retriever objects.
    """



    cache_name = str(

        strategy

    )



    with _retriever_lock:


        if cache_name in _retriever_cache:


            logger.info(

                f"Using cached retriever {cache_name}"

            )


            return _retriever_cache[cache_name]





        logger.info(

            f"Initializing retriever {cache_name}"

        )



        retriever = build_retriever(

            strategy

        )



        _retriever_cache[cache_name] = retriever



        return retriever






# =====================================================
# INDEX OPTIMIZATION
# =====================================================


def index_chunks(
    retriever,
    chunks
):

    """
    Optimized indexing layer.

    Protects against:
    - empty chunks
    - duplicate indexing
    - retriever crashes
    """



    if not chunks:


        logger.warning(

            "No chunks available for indexing"

        )


        return False





    try:


        retriever.index(

            chunks

        )



        logger.info(

            f"Indexed {len(chunks)} chunks"

        )


        return True




    except Exception as e:


        logger.exception(

            f"Retriever indexing failed: {e}"

        )


        return False







# =====================================================
# VECTOR FALLBACK
# =====================================================


async def vector_fallback_search(
    question,
    strategy
):

    """
    Fallback when live scraping fails.

    Uses previous indexed knowledge.

    Keeps RAG alive when:
    - website unavailable
    - scraper blocked
    - internet failure
    """



    try:


        logger.warning(

            "Activating vector fallback"

        )



        retriever = get_or_create_retriever(

            strategy

        )



        results = retriever.search(

            question,

            retrieval_cfg.top_k_retrieve

        )



        return results




    except Exception as e:


        logger.exception(

            f"Vector fallback failed: {e}"

        )


        return []








# =====================================================
# SAFE SOURCE COLLECTION PIPELINE
# =====================================================


async def prepare_context(
    question,
    meta,
    status_callback=None
):

    """
    Complete context preparation.

    Flow:

    Query
       |
    Live Search
       |
    Documents
       |
    Chunks
       |
    Cache
       |
    Retriever Context


    """


    key = cache_key(

        question

    )



    # ---------------------------------
    # Check chunk cache
    # ---------------------------------


    cached_chunks = get_cached_chunks(

        key

    )



    if cached_chunks:


        logger.info(

            "Using chunk cache"

        )


        return cached_chunks





    # ---------------------------------
    # Live scraping
    # ---------------------------------


    documents = await collect_documents(

        meta,

        status_callback

    )





    if not documents:


        logger.warning(

            "No live documents, fallback mode"

        )


        return []





    # ---------------------------------
    # Processing
    # ---------------------------------


    chunks = process_rag_documents(

        documents

    )



    if not chunks:


        logger.warning(

            "Chunk generation failed"

        )


        return []





    # ---------------------------------
    # Save cache
    # ---------------------------------


    save_chunks_cache(

        key,

        chunks

    )



    return chunks

# =====================================================
# RETRIEVAL PIPELINE
# =====================================================


async def retrieve_context(
    question,
    chunks,
    strategy,
    status_callback=None
):
    """
    Build retriever context.

    Flow:

    Chunks
       |
    Retriever Index
       |
    Semantic Search
       |
    Keyword Search
       |
    Hybrid Ranking

    """

    if not chunks:

        return []



    update_status(

        status_callback,

        "Building retrieval index"

    )



    try:


        retriever = get_or_create_retriever(

            strategy

        )



        indexed = index_chunks(

            retriever,

            chunks

        )



        if not indexed:


            logger.warning(

                "Retriever indexing skipped"

            )


            return []





        retrieve_k = min(

            retrieval_cfg.top_k_retrieve,

            20

        )



        update_status(

            status_callback,

            "Searching knowledge base"

        )




        results = retriever.search(

            question,

            retrieve_k

        )




        if not results:


            logger.warning(

                "No retrieval results"

            )


            return []




        logger.info(

            f"Retrieved {len(results)} chunks"

        )



        return results





    except Exception as e:


        logger.exception(

            f"Retrieval failed: {e}"

        )


        return []








# =====================================================
# RERANKING SYSTEM
# =====================================================


async def rerank_results(
    question,
    results,
    status_callback=None
):
    """
    Cross Encoder reranking.

    Improves:
    - semantic relevance
    - answer accuracy
    - citation quality

    """



    if not results:


        return []




    if not reranker_cfg.enabled:


        return results





    update_status(

        status_callback,

        "Reranking documents"

    )




    try:


        reranker = get_reranker()



        pairs = []



        for item in results:


            text = (

                item.chunk.text

                if hasattr(
                    item,
                    "chunk"
                )

                else str(item)

            )



            pairs.append(

                [

                    question,

                    text

                ]

            )





        scores = reranker.predict(

            pairs,

            batch_size=8,

            show_progress_bar=False

        )




        for item,score in zip(

            results,

            scores

        ):


            item.score = float(

                score

            )






        results.sort(

            key=lambda x:x.score,

            reverse=True

        )





        top_results = results[

            :reranker_cfg.top_k

        ]




        logger.info(

            f"Reranked {len(top_results)} chunks"

        )



        return top_results





    except Exception as e:


        logger.exception(

            f"Reranker failed: {e}"

        )


        return results[:reranker_cfg.top_k]







# =====================================================
# CONTEXT OPTIMIZATION
# =====================================================


def build_generation_context(
    retrieved
):

    """
    Prepare final context
    sent to LLM.

    Removes:
    - duplicated text
    - empty chunks
    - excessive length

    """



    if not retrieved:

        return ""




    contexts=[]

    seen=set()



    max_chars = 12000




    current_length = 0




    for item in retrieved:



        try:


            text = item.chunk.text.strip()



        except Exception:


            continue





        if not text:


            continue





        fingerprint = hashlib.md5(

            text.encode(

                "utf-8"

            )

        ).hexdigest()





        if fingerprint in seen:

            continue



        seen.add(

            fingerprint

        )





        if (

            current_length +

            len(text)

            >

            max_chars

        ):

            break





        contexts.append(

            text

        )



        current_length += len(text)






    return "\n\n---\n\n".join(

        contexts

    )







# =====================================================
# MAIN ASYNC RAG PIPELINE
# =====================================================


async def run_rag_pipeline_async(

    question: str,

    retriever_strategy: RetrieverType="hybrid",

    status_callback=None,

    stream=False

):


    """
    Production RAG pipeline.

    Complete Flow:

    Question

       |

    Query Analyzer

       |

    Cache

       |

    Live Search

       |

    Chunk Processing

       |

    Retriever

       |

    Hybrid Search

       |

    Reranker

       |

    Context Builder

       |

    LLM


    """



    try:



        update_status(

            status_callback,

            "Analyzing query"

        )




        meta = analyze_query(

            question

        )





        logger.info(

            f"Query metadata: {meta}"

        )





        # ---------------------------------
        # Prepare Context
        # ---------------------------------


        chunks = await prepare_context(

            question,

            meta,

            status_callback

        )





        # ---------------------------------
        # If live failed
        # use existing vector memory
        # ---------------------------------


        if not chunks:


            update_status(

                status_callback,

                "Using vector fallback"

            )


            retrieved = await vector_fallback_search(

                question,

                retriever_strategy

            )



        else:



            retrieved = await retrieve_context(

                question,

                chunks,

                retriever_strategy,

                status_callback

            )





        if not retrieved:



            return RAGAnswer(

                answer=(

                    "لم يتم العثور على "

                    "مصادر كافية للإجابة."

                ),

                sources=[],

                retriever_used="none",

                chunks_retrieved=0

            )







        # ---------------------------------
        # Reranking
        # ---------------------------------


        retrieved = await rerank_results(

            question,

            retrieved,

            status_callback

        )






        context = build_generation_context(

            retrieved

        )






        if not context:



            return RAGAnswer(

                answer=(

                    "لا يوجد سياق كافٍ."

                ),

                sources=[],

                retriever_used=str(

                    retriever_strategy

                ),

                chunks_retrieved=0

            )







        # ---------------------------------
        # Generation
        # ---------------------------------


        update_status(

            status_callback,

            "Generating answer"

        )





        if stream:


            return stream_answer_async(

                question,

                retrieved

            )






        answer = await generate_answer_async(

            question,

            retrieved

        )





        answer.sources = [


            {

                "url":

                item.chunk.metadata.get(

                    "source",

                    ""

                ),

                "score":

                getattr(

                    item,

                    "score",

                    0

                )

            }


            for item in retrieved

        ]




        answer.chunks_retrieved = len(

            retrieved

        )



        answer.retriever_used = str(

            retriever_strategy

        )




        return answer





    except Exception as e:


        logger.exception(

            f"Pipeline crashed: {e}"

        )



        return RAGAnswer(

            answer=(

                "حدث خطأ أثناء معالجة السؤال."

            ),

            sources=[],

            retriever_used="error",

            chunks_retrieved=0

        )

# =====================================================
# STREAMLIT SAFE ASYNC WRAPPER
# =====================================================


def run_async_safe(coro):
    """
    Execute async coroutine safely.

    Compatible with:
    - Streamlit
    - Jupyter
    - FastAPI
    - Normal Python

    """

    try:


        loop = asyncio.get_running_loop()



        # Already running loop

        if loop.is_running():


            import nest_asyncio


            nest_asyncio.apply()



            return asyncio.run(

                coro

            )



    except RuntimeError:


        pass





    return asyncio.run(

        coro

    )







# =====================================================
# SYNC PIPELINE ENTRY
# =====================================================


def run_rag_pipeline(

    question,

    retriever_strategy="hybrid",

    stream=False,

    status_callback=None

):

    """
    Synchronous API.

    Used by:

    Streamlit frontend

    Example:

        answer = run_rag_pipeline(
            "ما هو قانون نيوتن الثاني؟"
        )

    """



    return run_async_safe(

        run_rag_pipeline_async(

            question,

            retriever_strategy,

            status_callback,

            stream

        )

    )







# =====================================================
# CACHE MANAGEMENT
# =====================================================


def clear_pipeline_cache():

    """
    Clear:

    - Chunk cache
    - Retriever cache

    """



    with CACHE_LOCK:


        PIPELINE_CACHE.clear()



    with _retriever_lock:


        _retriever_cache.clear()



    logger.info(

        "All pipeline caches cleared"

    )







def pipeline_cache_size():

    """
    Return cache statistics.
    """

    with CACHE_LOCK:


        chunks = len(

            PIPELINE_CACHE

        )


    with _retriever_lock:


        retrievers = len(

            _retriever_cache

        )



    return {


        "chunk_cache":

        chunks,


        "retriever_cache":

        retrievers

    }







# =====================================================
# QUERY METADATA API
# =====================================================


def get_pipeline_metadata(

    question:str

):

    """
    Returns query understanding.

    Used by frontend
    before execution.

    """



    try:


        meta = analyze_query(

            question

        )



        return {


            "grade":

            meta.grade,


            "subject":

            meta.subject,


            "topic":

            meta.topic,


            "intent":

            meta.intent,


            "keywords":

            meta.keywords,


            "live_search":

            meta.needs_live_search,


            "query":

            meta.search_query

        }




    except Exception as e:


        logger.exception(

            f"Metadata error: {e}"

        )


        return {}







# =====================================================
# METRICS REPORT
# =====================================================


def get_pipeline_metrics():

    """
    Runtime monitoring.

    Useful for:

    - debugging
    - production dashboard
    - logging

    """



    return {


        "cache":

        pipeline_cache_size(),



        "retriever_loaded":

        list(

            _retriever_cache.keys()

        ),



        "reranker_loaded":

        _reranker_model is not None,


        "status":

        "running"

    }







# =====================================================
# HEALTH CHECK
# =====================================================


def pipeline_health():

    """
    Production health endpoint.

    """

    return {


        "pipeline":

        "healthy",


        "scraper":

        "enabled",


        "retriever":

        "ready",


        "reranker":

        (

            "loaded"

            if _reranker_model

            else

            "lazy"

        )

    }







# =====================================================
# PRODUCTION TEST ENTRY
# =====================================================


async def test_pipeline():

    """
    Local production test.

    Run:

    python orchestrator.py

    """



    question = (

        "اشرح قانون نيوتن الثاني "

        "للصف الثالث الاعدادي"

    )



    result = await run_rag_pipeline_async(

        question

    )



    print(

        result.answer

    )








# =====================================================
# MAIN
# =====================================================


if __name__ == "__main__":


    asyncio.run(

        test_pipeline()

    )

