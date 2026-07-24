"""
Production Real-Time RAG Orchestrator v6

Responsibilities:
- Query analysis
- Live retrieval
- Cache management
- Retriever lifecycle
- Reranking
- LLM orchestration

Compatible:
- live_scraper.py
- query_analyzer.py
- chunker.py
- retriever.py
- rag.chain
"""

from __future__ import annotations


# =====================================================
# Standard Library
# =====================================================

import asyncio

import hashlib

import time

from dataclasses import dataclass

from threading import Lock, Thread



# =====================================================
# Third Party
# =====================================================

from cachetools import TTLCache

from loguru import logger



# =====================================================
# Internal Imports
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


@dataclass(frozen=True)
class PipelineConfig:

    """
    Global production configuration.
    """


    cache_size: int = 200


    cache_ttl: int = 600


    max_retrieve: int = 20


    max_context_chars: int = 12000


    scraper_timeout: int = 60


    version: str = "rag-v6"





PIPELINE_CONFIG = PipelineConfig()






# =====================================================
# Cache Key
# =====================================================


def cache_key(question: str) -> str:

    """
    Generate stable cache key.
    """

    if not question:

        return "empty"


    normalized = (
        question
        .strip()
        .lower()
    )


    return hashlib.sha256(

        normalized.encode(
            "utf-8"
        )

    ).hexdigest()






# =====================================================
# Thread Safe Pipeline Cache
# =====================================================


class PipelineCache:


    """
    Stores processed chunks.

    Thread safe for:
    - Streamlit
    - Multi request usage
    """


    def __init__(self):


        self.cache = TTLCache(

            maxsize=PIPELINE_CONFIG.cache_size,

            ttl=PIPELINE_CONFIG.cache_ttl

        )


        self.lock = Lock()





    def get(
        self,
        key: str
    ):


        with self.lock:

            return self.cache.get(

                key

            )







    def set(
        self,
        key: str,
        value
    ):


        with self.lock:

            self.cache[key] = value







    def clear(self):


        with self.lock:

            self.cache.clear()


        logger.info(
            "Pipeline cache cleared"
        )







    def size(self):


        with self.lock:

            return len(
                self.cache
            )






PIPELINE_CACHE = PipelineCache()






# =====================================================
# Retriever Manager
# =====================================================


class RetrieverManager:


    """
    Manage retriever lifecycle.

    Prevent rebuilding vector index
    repeatedly.
    """



    def __init__(self):


        self.instances = {}

        self.lock = Lock()






    def get(
        self,
        strategy: RetrieverType
    ):


        key = str(strategy)



        with self.lock:


            if key not in self.instances:


                logger.info(

                    f"Creating retriever: {key}"

                )


                self.instances[key] = build_retriever(

                    strategy

                )



            return self.instances[key]







    def clear(self):


        with self.lock:

            self.instances.clear()


        logger.info(
            "Retriever cache cleared"
        )






    def count(self):


        with self.lock:

            return len(
                self.instances
            )







RETRIEVER_MANAGER = RetrieverManager()






# =====================================================
# Reranker Manager
# =====================================================


_reranker_model = None


_reranker_lock = Lock()





def get_reranker():


    global _reranker_model



    if not reranker_cfg.enabled:

        return None





    with _reranker_lock:


        if _reranker_model is None:


            logger.info(

                "Loading CrossEncoder reranker"

            )


            from sentence_transformers import CrossEncoder



            _reranker_model = CrossEncoder(

                reranker_cfg.model,

                max_length=512

            )



        return _reranker_model







# =====================================================
# Status Helper
# =====================================================


def update_status(
    callback,
    message: str
):


    logger.info(
        message
    )


    if callback:


        try:

            callback(
                message
            )

        except Exception:


            logger.warning(

                "Status callback failed"

            )







# =====================================================
# Async Safe Runner
# =====================================================


def run_async_safe(
    coroutine
):

    """
    Streamlit compatible async executor.
    """


    try:

        loop = asyncio.get_running_loop()



        if loop.is_running():


            result = []

            errors = []



            def runner():


                try:


                    new_loop = asyncio.new_event_loop()


                    asyncio.set_event_loop(

                        new_loop

                    )


                    result.append(

                        new_loop.run_until_complete(

                            coroutine

                        )

                    )



                except Exception as e:


                    errors.append(e)



                finally:


                    new_loop.close()





            thread = Thread(

                target=runner

            )


            thread.start()


            thread.join()



            if errors:

                raise errors[0]



            return result[0]



    except RuntimeError:


        pass





    return asyncio.run(

        coroutine

    )







# =====================================================
# Cache Management API
# =====================================================


def clear_pipeline_cache():


    PIPELINE_CACHE.clear()


    RETRIEVER_MANAGER.clear()




def pipeline_cache_stats():


    return {


        "chunks":

            PIPELINE_CACHE.size(),


        "retrievers":

            RETRIEVER_MANAGER.count(),


        "reranker_loaded":

            _reranker_model is not None

    }
# =====================================================
# LIVE SEARCH COLLECTION
# =====================================================


async def collect_documents(
    meta: QueryMetadata,
    status_callback=None
):
    """
    Collect documents from live scraper.

    Failure safe:
    - scraper crash does not stop RAG
    - returns empty list
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
                "No documents returned from scraper"
            )


            return []



        logger.info(

            f"Scraper returned {len(documents)} documents"

        )


        return documents




    except Exception as e:


        logger.exception(

            f"Live search failed: {e}"

        )


        return []






# =====================================================
# DOCUMENT PROCESSING
# =====================================================


def process_rag_documents(
    documents
):
    """
    Convert documents into RAG chunks.

    Includes:
    - chunk generation
    - duplicate filtering
    - validation
    """


    if not documents:

        return []



    try:


        chunks = process_documents(

            documents

        )



        if not chunks:


            logger.warning(

                "Chunk processor returned empty result"

            )


            return []





        unique = {}



        for chunk in chunks:



            try:


                text = chunk.text.strip()



            except Exception:


                continue





            if not text:


                continue





            fingerprint = hashlib.md5(

                text.encode(

                    "utf-8"

                )

            ).hexdigest()





            if fingerprint not in unique:


                unique[fingerprint] = chunk





        final_chunks = list(

            unique.values()

        )





        logger.info(

            f"""

Document Processing:

Documents:
{len(documents)}

Chunks:
{len(chunks)}

Unique:
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
# CHUNK CACHE OPERATIONS
# =====================================================


def get_cached_chunks(
    question: str
):
    """
    Retrieve cached chunks.
    """


    key = cache_key(

        question

    )


    cached = PIPELINE_CACHE.get(

        key

    )



    if cached:


        logger.info(

            "Using cached chunks"

        )



    return cached







def save_chunks_cache(
    question: str,
    chunks
):
    """
    Store processed chunks.
    """


    if not chunks:

        return



    key = cache_key(

        question

    )



    PIPELINE_CACHE.set(

        key,

        chunks

    )



    logger.info(

        f"Cached {len(chunks)} chunks"

    )








# =====================================================
# PREPARE RAG CONTEXT
# =====================================================


async def prepare_context(
    question: str,
    meta: QueryMetadata,
    status_callback=None
):
    """
    Complete context preparation.

    Flow:

    Question
        |
    Cache
        |
    Live Search
        |
    Documents
        |
    Chunking
        |
    Cache Save

    """



    # -----------------------------
    # Cache check
    # -----------------------------


    cached = get_cached_chunks(

        question

    )



    if cached:


        return cached






    # -----------------------------
    # Live Search
    # -----------------------------


    documents = await collect_documents(

        meta,

        status_callback

    )



    if not documents:


        logger.warning(

            "No documents available"

        )


        return []






    # -----------------------------
    # Processing
    # -----------------------------


    update_status(

        status_callback,

        "Processing documents"

    )



    chunks = process_rag_documents(

        documents

    )




    if not chunks:


        return []






    # -----------------------------
    # Cache save
    # -----------------------------


    save_chunks_cache(

        question,

        chunks

    )



    return chunks







# =====================================================
# RETRIEVER INDEXING
# =====================================================


def index_chunks(
    retriever,
    chunks
):
    """
    Index chunks safely.
    """


    if not chunks:


        logger.warning(

            "No chunks to index"

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

            f"Indexing failed: {e}"

        )


        return False








# =====================================================
# RETRIEVAL LAYER
# =====================================================


async def retrieve_context(
    question: str,
    chunks,
    strategy: RetrieverType,
    status_callback=None
):
    """
    Retrieve relevant chunks.

    Supports:
    - vector search
    - hybrid retrieval
    """



    if not chunks:


        return []





    try:


        update_status(

            status_callback,

            "Building retriever"

        )




        retriever = RETRIEVER_MANAGER.get(

            strategy

        )





        indexed = index_chunks(

            retriever,

            chunks

        )



        if not indexed:


            return []







        update_status(

            status_callback,

            "Searching knowledge base"

        )





        top_k = min(

            retrieval_cfg.top_k_retrieve,

            PIPELINE_CONFIG.max_retrieve

        )






        results = retriever.search(

            question,

            top_k

        )





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
# VECTOR FALLBACK
# =====================================================


async def vector_fallback_search(
    question: str,
    strategy: RetrieverType
):
    """
    Use existing retriever memory
    if live search fails.
    """


    try:


        logger.warning(

            "Using vector fallback"

        )



        retriever = RETRIEVER_MANAGER.get(

            strategy

        )



        return retriever.search(

            question,

            retrieval_cfg.top_k_retrieve

        )



    except Exception as e:


        logger.exception(

            f"Vector fallback failed: {e}"

        )


        return []

# =====================================================
# RERANKING SYSTEM
# =====================================================


async def rerank_results(
    question: str,
    results,
    status_callback=None
):
    """
    Cross Encoder reranking.

    Improves:
    - semantic relevance
    - answer quality
    - source selection
    """


    if not results:

        return []



    if not reranker_cfg.enabled:


        return results






    try:


        update_status(

            status_callback,

            "Reranking results"

        )



        reranker = get_reranker()



        if not reranker:


            return results





        pairs = []



        for item in results:


            try:


                text = item.chunk.text


            except Exception:


                text = ""



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







        for item, score in zip(

            results,

            scores

        ):


            item.score = float(

                score

            )







        results.sort(

            key=lambda x: x.score,

            reverse=True

        )






        final_results = results[

            :reranker_cfg.top_k

        ]




        logger.info(

            f"Reranking finished: {len(final_results)} chunks"

        )



        return final_results






    except Exception as e:


        logger.exception(

            f"Reranker error: {e}"

        )



        return results[:reranker_cfg.top_k]








# =====================================================
# CONTEXT OPTIMIZATION
# =====================================================


def build_generation_context(
    retrieved
):
    """
    Prepare clean context for LLM.

    Removes:
    - duplicate chunks
    - empty text
    - oversized context
    """



    if not retrieved:


        return ""




    contexts = []

    seen = set()


    current_size = 0



    max_size = PIPELINE_CONFIG.max_context_chars






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

            current_size + len(text)

            >

            max_size

        ):


            break





        contexts.append(

            text

        )


        current_size += len(text)








    return "\n\n---\n\n".join(

        contexts

    )









# =====================================================
# SOURCE BUILDER
# =====================================================


def build_sources(
    retrieved
):
    """
    Prepare citation metadata.
    """



    sources = []



    for item in retrieved:



        try:


            chunk = item.chunk



            sources.append(

                {

                    "title":

                        getattr(

                            chunk,

                            "title",

                            ""

                        ),


                    "url":

                        getattr(

                            chunk,

                            "source_url",

                            ""

                        ),


                    "score":

                        round(

                            float(

                                getattr(

                                    item,

                                    "score",

                                    0

                                )

                            ),

                            4

                        )

                }

            )



        except Exception:


            continue




    return sources








# =====================================================
# MAIN ASYNC RAG PIPELINE
# =====================================================


async def run_rag_pipeline_async(
    question: str,
    retriever_strategy: RetrieverType = "hybrid",
    status_callback=None,
    stream=False
):
    """
    Production RAG execution.

    Flow:

    Question
        |
    Analyzer
        |
    Live Search
        |
    Chunk Cache
        |
    Retriever
        |
    Reranker
        |
    LLM

    """



    try:



        # -----------------------------
        # Query Analysis
        # -----------------------------


        update_status(

            status_callback,

            "Analyzing query"

        )




        meta = analyze_query(

            question

        )





        logger.info(

            {

                "query":

                    question,

                "intent":

                    meta.intent,

                "subject":

                    meta.subject,

                "grade":

                    meta.grade,

                "confidence":

                    meta.confidence

            }

        )







        # -----------------------------
        # Context Preparation
        # -----------------------------


        chunks = await prepare_context(

            question,

            meta,

            status_callback

        )






        # -----------------------------
        # Retrieval
        # -----------------------------


        if chunks:


            retrieved = await retrieve_context(

                question,

                chunks,

                retriever_strategy,

                status_callback

            )



        else:


            retrieved = await vector_fallback_search(

                question,

                retriever_strategy

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








        # -----------------------------
        # Reranking
        # -----------------------------


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

                answer="لا يوجد سياق مناسب.",

                sources=[],

                retriever_used=str(

                    retriever_strategy

                ),

                chunks_retrieved=0

            )








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







        answer = await generate_answer_async(

            question,

            retrieved

        )







        answer.sources = build_sources(

            retrieved

        )



        answer.retriever_used = str(

            retriever_strategy

        )


        answer.chunks_retrieved = len(

            retrieved

        )





        return answer







    except Exception as e:


        logger.exception(

            f"RAG pipeline crashed: {e}"

        )



        return RAGAnswer(

            answer=(

                "حدث خطأ أثناء تشغيل نظام الإجابة."

            ),

            sources=[],

            retriever_used="error",

            chunks_retrieved=0

        )

# =====================================================
# STREAMLIT SAFE SYNC WRAPPER
# =====================================================


def run_rag_pipeline(
    question: str,
    retriever_strategy="hybrid",
    stream=False,
    status_callback=None
):
    """
    Public synchronous API.

    Used by:
    - Streamlit frontend
    - API endpoints
    - CLI testing
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
# CACHE CLEAR API
# =====================================================


def clear_pipeline_cache():

    """
    Clear all runtime caches.
    """


    PIPELINE_CACHE.clear()


    RETRIEVER_MANAGER.clear()


    logger.info(

        "All pipeline resources cleared"

    )







# =====================================================
# PIPELINE METADATA
# =====================================================


def get_pipeline_metadata(
    question: str
):
    """
    Returns query understanding
    before running pipeline.
    """



    try:


        meta = analyze_query(

            question

        )



        return {


            "query":

                question,


            "normalized":

                meta.normalized,


            "intent":

                meta.intent,


            "subject":

                meta.subject,


            "stage":

                meta.stage,


            "grade":

                meta.grade,


            "topic":

                meta.topic,


            "keywords":

                meta.keywords,


            "search_query":

                meta.search_query,


            "source_category":

                meta.source_category,


            "live_search":

                meta.needs_live_search,


            "confidence":

                meta.confidence

        }





    except Exception as e:


        logger.exception(

            f"Metadata generation failed: {e}"

        )


        return {

            "error":

                str(e)

        }









# =====================================================
# PIPELINE METRICS
# =====================================================


_PIPELINE_START_TIME = time.time()





def get_pipeline_metrics():
    """
    Runtime monitoring.

    Used for:
    - debugging
    - dashboard
    - health monitoring
    """



    return {


        "version":

            PIPELINE_CONFIG.version,


        "uptime_seconds":

            round(

                time.time()

                -

                _PIPELINE_START_TIME,

                2

            ),



        "cache":


            pipeline_cache_stats(),




        "retrievers_loaded":

            RETRIEVER_MANAGER.count(),



        "reranker_loaded":

            _reranker_model is not None,



        "status":

            "healthy"

    }









# =====================================================
# HEALTH CHECK
# =====================================================


def pipeline_health():
    """
    Production health status.
    """



    return {


        "pipeline":

            "online",



        "version":

            PIPELINE_CONFIG.version,



        "cache":

            "ready",



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
# TEST FUNCTION
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



    print(

        result.sources

    )









# =====================================================
# PRODUCTION ENTRY POINT
# =====================================================


if __name__ == "__main__":


    asyncio.run(

        test_pipeline()

    )



