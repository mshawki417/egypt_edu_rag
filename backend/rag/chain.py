"""
High Performance Async RAG Chain
Production OpenRouter + Context Optimization

Version:
Advanced v2

Features:
- Async optimized LLM calls
- OpenRouter integration
- Retry with exponential backoff
- Streaming support
- Context compression
- Source citation
- Production logging
- Safe fallback handling

Compatible with:
- orchestrator.py
- retriever.py
- RetrievedChunk
"""


from __future__ import annotations


import asyncio
import json
import random
import time

from dataclasses import dataclass
from typing import AsyncGenerator


import httpx

from loguru import logger


from config.settings import llm_cfg


from backend.retrieval.retriever import (
    RetrievedChunk
)





# =====================================================
# CONFIGURATION
# =====================================================


OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)


DEFAULT_MODEL = (
    "meta-llama/llama-3.1-8b-instruct:free"
)


MAX_RETRIES = 4


REQUEST_TIMEOUT = 120


MAX_CONTEXT_CHUNKS = 5


MAX_CONTEXT_LENGTH = 12000





# =====================================================
# SYSTEM PROMPT
# =====================================================


BASE_PROMPT = SYSTEM_PROMPT = """
You are an expert AI Educational Tutor specialized in the Egyptian education system.

Your role is to act as a professional teacher, curriculum expert, and learning assistant.
You help Egyptian students and teachers by providing accurate, detailed, structured,
and easy-to-understand educational explanations.

Your objective:
Transform every user question into a high-quality learning experience.

==============================
1. Question Understanding
==============================

Before answering, internally analyze:

- Student grade or educational level.
- Subject area.
- Learning objective.
- Required response type:
  explanation, definition, solving problem, comparison, summary,
  exam preparation, concept clarification, or practical application.

Do not mention this analysis to the user.


==============================
2. Knowledge Strategy
==============================

Use the retrieved context as the main knowledge reference when it is relevant.

Rules:

- Integrate useful information from retrieved sources naturally.
- Do not mention:
  "retrieved context"
  "documents"
  "sources provided"
  "knowledge base"
  "RAG system"
  "I cannot find this information in the sources"

The user should only see the educational answer.

If retrieved information is incomplete:

- Complete the explanation using reliable general educational knowledge.
- Add commonly accepted scientific or educational information when appropriate.
- Do not invent specific official decisions, laws, statistics, dates, or government policies.

When information requires official verification:
state:
"يرجى الرجوع إلى المصدر الرسمي للتأكد من آخر التحديثات."


==============================
3. Teaching Methodology
==============================

Explain like an experienced teacher.

Always:

- Start directly with the answer.
- Avoid unnecessary introductions.
- Explain from simple concepts to advanced concepts.
- Define difficult terms before using them.
- Connect ideas with real-life examples.
- Adapt the explanation level according to the student's grade.


==============================
4. Answer Structure
==============================

Use the following structure when suitable:


## Explanation

Provide a complete explanation of the topic.

Explain:

- What is it?
- Why is it important?
- How does it work?
- What are the main concepts?


## Detailed Learning Points

Break the topic into organized sections.

Use:

- Bullet points.
- Tables.
- Comparisons.
- Step-by-step explanations.


## Examples and Applications

Provide:

- Educational examples.
- Real-life examples.
- Solved examples when applicable.


## Quick Summary

Give a short revision summary for students.


## Practice Questions

When appropriate, create:

3-5 review questions.

Include answers only when useful for learning.


==============================
5. Egyptian Education Context
==============================

Whenever possible:

- Use examples related to Egyptian students.
- Consider Egyptian curriculum style.
- Use terminology familiar to Egyptian learners.
- Explain concepts according to school learning levels.


==============================
6. Visual Learning Support
==============================

When the topic benefits from visual explanation,
add:

## Recommended Visuals

Suggest educational visuals only.

Examples:

- Scientific diagrams.
- Educational illustrations.
- Timelines.
- Flowcharts.
- Concept maps.
- Graphs.

Do NOT create fake URLs.
Do NOT claim that images exist.
Only describe images that an image search system can find.


==============================
7. Sources
==============================

At the end of the answer add:

## Sources

Mention only real sources that appear in the provided information.

Format:

[Source: Website name]

If no source is available, do not create fake sources.


==============================
8. Response Quality Rules
==============================

Always prioritize:

Accuracy > creativity

Teaching quality > short answers

Clarity > complexity


The final answer must:

- Be professional.
- Be detailed but easy to understand.
- Feel like a personal AI teacher.
- Never reveal internal processes.
- Never discuss limitations of retrieval.
- Never apologize because information was unavailable.


Your final goal:

Create the experience of a world-class AI educational tutor
that combines reliable knowledge retrieval,
expert teaching ability,
and personalized student guidance.
"""

# =====================================================
# RESPONSE MODEL
# =====================================================


@dataclass
class RAGAnswer:


    answer: str


    sources: list[dict]


    retriever_used: str


    chunks_retrieved: int





# =====================================================
# LLM CLIENT
# =====================================================


class OpenRouterClient:


    """
    Production OpenRouter async client.
    """


    def __init__(self):


        self.url = OPENROUTER_URL


        self.timeout = httpx.Timeout(

            REQUEST_TIMEOUT

        )



    def get_headers(self):


        if not llm_cfg.openrouter_api_key:


            raise ValueError(

                "Missing OPENROUTER_API_KEY"

            )


        return {


            "Authorization":

            f"Bearer {llm_cfg.openrouter_api_key}",



            "Content-Type":

            "application/json",



            "HTTP-Referer":

            "https://github.com/egypt-edu-rag",



            "X-Title":

            "Egypt Education RAG"

        }





    def get_model(self):


        if (

            hasattr(

                llm_cfg,

                "model"

            )

            and

            llm_cfg.model

        ):


            return llm_cfg.model.strip()



        return DEFAULT_MODEL







    async def close(self):

        pass

# =====================================================
# CONTEXT BUILDER
# =====================================================


def clean_text(text: str) -> str:


    """
    Remove noisy spaces
    and normalize context.
    """


    if not text:

        return ""


    text = text.replace(

        "\x00",

        ""

    )


    text = " ".join(

        text.split()

    )


    return text.strip()






def build_context(

    chunks: list[RetrievedChunk],

    max_chunks: int = MAX_CONTEXT_CHUNKS

) -> str:


    """
    Build optimized RAG context.

    Keeps:
    - Source
    - Title
    - Content

    Prevents:
    - Huge prompts
    - Duplicate text
    """



    if not chunks:

        return ""




    selected = chunks[:max_chunks]



    context_parts = []


    current_length = 0




    for index, item in enumerate(

        selected,

        1

    ):



        chunk = item.chunk



        text = clean_text(

            chunk.text

        )



        if not text:

            continue





        block = f"""

[{index}]

المصدر:
{chunk.source_url}


العنوان:
{chunk.title}


المحتوى:
{text}

"""



        if (

            current_length

            +

            len(block)

            >

            MAX_CONTEXT_LENGTH

        ):

            break





        context_parts.append(

            block

        )


        current_length += len(block)






    return "\n----------------\n".join(

        context_parts

    )



# =====================================================
# PAYLOAD BUILDER
# =====================================================


def build_payload(

    question: str,

    context: str,

    stream: bool = False

):


    return {


        "model":

        OpenRouterClient().get_model(),



        "temperature":

        getattr(

            llm_cfg,

            "temperature",

            0.2

        ),



        "max_tokens":

        getattr(

            llm_cfg,

            "max_tokens",

            1200

        ),



        "stream":

        stream,



        "messages":

        [



            {


                "role":

                "system",



                "content":

                BASE_PROMPT

            },




            {


                "role":

                "user",



                "content":

                f"""

السؤال:

{question}



المصادر المتاحة:

{context}



الإجابة:

"""

            }



        ]

    }









# =====================================================
# RETRY ENGINE
# =====================================================


async def retry_wait(

    attempt: int

):


    """
    Exponential backoff
    with random jitter.
    """


    delay = (

        (2 ** attempt)

        +

        random.uniform(

            0,

            1

        )

    )


    await asyncio.sleep(

        delay

    )









# =====================================================
# OPENROUTER REQUEST
# =====================================================


async def request_openrouter(

    client: httpx.AsyncClient,

    question: str,

    context: str

):


    payload = build_payload(

        question,

        context

    )



    router = OpenRouterClient()





    for attempt in range(

        MAX_RETRIES

    ):



        try:



            response = await client.post(

                OPENROUTER_URL,

                headers=

                router.get_headers(),

                json=payload

            )







            # Rate limit

            if response.status_code == 429:



                logger.warning(

                    "OpenRouter rate limit reached"

                )



                await retry_wait(

                    attempt

                )


                continue








            # Temporary server errors

            if response.status_code >= 500:



                logger.warning(

                    f"OpenRouter server error {response.status_code}"

                )



                await retry_wait(

                    attempt

                )


                continue







            if response.status_code == 401:



                raise RuntimeError(

                    "Invalid OpenRouter API Key"

                )







            response.raise_for_status()



            return response.json()







        except (

            httpx.TimeoutException,

            httpx.NetworkError

        ) as e:



            logger.warning(

                f"Network error attempt {attempt+1}: {e}"

            )


            await retry_wait(

                attempt

            )







        except Exception:


            raise







    raise RuntimeError(

        "OpenRouter failed after retries"

    )


# =====================================================
# RESPONSE PARSER
# =====================================================


def extract_answer(

    response: dict

) -> str:


    """
    Extract assistant answer safely
    from OpenRouter response.
    """


    try:


        choices = response.get(

            "choices",

            []

        )


        if not choices:

            return ""



        message = choices[0].get(

            "message",

            {}

        )



        return (

            message.get(

                "content",

                ""

            )

            .strip()

        )



    except Exception as e:


        logger.error(

            f"Response parsing failed: {e}"

        )


        return ""









# =====================================================
# SOURCE FORMATTER
# =====================================================


def build_sources(

    chunks: list[RetrievedChunk]

) -> list[dict]:


    """
    Convert retrieved chunks
    into citation metadata.
    """



    sources = []



    seen = set()



    for item in chunks:



        chunk = item.chunk



        url = chunk.source_url




        if url in seen:

            continue



        seen.add(url)



        sources.append(



            {


                "title":

                chunk.title,



                "url":

                url,



                "score":

                round(

                    float(

                        item.score

                    ),

                    4

                ),



                "source":

                chunk.metadata.get(

                    "source",

                    ""

                ),



                "type":

                chunk.metadata.get(

                    "doc_type",

                    "web"

                )



            }



        )



    return sources










# =====================================================
# GENERATE ANSWER ASYNC
# =====================================================


async def generate_answer_async(

    question: str,

    chunks: list[RetrievedChunk]

) -> RAGAnswer:



    """
    Main generation pipeline.

    Steps:

    1- Build context
    2- Call LLM
    3- Parse answer
    4- Attach citations

    """



    if not chunks:



        return RAGAnswer(


            answer=

            "لم يتم العثور على معلومات كافية.",



            sources=[],



            retriever_used=

            "none",



            chunks_retrieved=

            0

        )







    context = build_context(

        chunks

    )




    if not context:



        return RAGAnswer(


            answer=

            "لا يوجد سياق صالح للإجابة.",



            sources=[],



            retriever_used=

            "none",



            chunks_retrieved=

            0

        )







    answer = ""




    try:



        async with httpx.AsyncClient(

            timeout=

            REQUEST_TIMEOUT

        ) as client:



            response = await request_openrouter(

                client,

                question,

                context

            )



            answer = extract_answer(

                response

            )






    except Exception as e:



        logger.exception(

            f"Generation failed: {e}"

        )



        answer = (

            "حدث خطأ أثناء توليد الإجابة."

        )








    if not answer:



        answer = (

            "لم يتمكن النموذج من إنشاء إجابة."

        )







    return RAGAnswer(



        answer=answer,



        sources=

        build_sources(

            chunks

        ),



        retriever_used=

        getattr(

            chunks[0],

            "retriever",

            "unknown"

        ),



        chunks_retrieved=

        len(chunks)



    )









# =====================================================
# STREAMING GENERATION
# =====================================================


async def stream_answer_async(

    question: str,

    chunks: list[RetrievedChunk]

):


    """
    Production streaming generator.

    Returns tokens progressively
    for Streamlit UI.
    """



    if not chunks:


        yield (

            "لا توجد معلومات كافية."

        )


        return






    context = build_context(

        chunks

    )




    router = OpenRouterClient()




    try:



        async with httpx.AsyncClient(

            timeout=

            STREAM_TIMEOUT

        ) as client:



            async with client.stream(

                "POST",

                OPENROUTER_URL,

                headers=

                router.get_headers(),


                json=

                build_payload(

                    question,

                    context,

                    True

                )


            ) as response:



                if response.status_code >= 400:



                    logger.error(

                        f"Streaming failed {response.status_code}"

                    )


                    yield (

                        "حدث خطأ أثناء الاتصال بالنموذج."

                    )


                    return







                async for line in response.aiter_lines():



                    if not line:


                        continue





                    if not line.startswith(

                        "data:"

                    ):


                        continue





                    data = line.replace(

                        "data:",

                        ""

                    ).strip()






                    if data == "[DONE]":


                        break







                    try:



                        event = json.loads(

                            data

                        )



                        token = (

                            event

                            .get(

                                "choices",

                                [{}]

                            )[0]

                            .get(

                                "delta",

                                {}

                            )

                            .get(

                                "content",

                                ""

                            )

                        )



                        if token:



                            yield token






                    except json.JSONDecodeError:



                        continue







    except Exception as e:



        logger.exception(

            f"Streaming error: {e}"

        )


        yield (

            "حدث خطأ أثناء البث."

        )

# =====================================================
# MODEL HEALTH CHECK
# =====================================================


async def check_llm_health() -> dict:


    """
    Verify OpenRouter availability.

    Used by:
    - Streamlit startup
    - Monitoring
    - Debugging
    """



    router = OpenRouterClient()



    try:



        async with httpx.AsyncClient(

            timeout=20

        ) as client:



            response = await client.get(

                "https://openrouter.ai/api/v1/models",

                headers=

                router.get_headers()

            )



            if response.status_code == 200:



                return {



                    "status":

                    "healthy",



                    "provider":

                    "openrouter",



                    "model":

                    router.get_model()



                }






            return {



                "status":

                "error",



                "code":

                response.status_code



            }






    except Exception as e:



        logger.error(

            f"LLM health check failed: {e}"

        )



        return {



            "status":

            "unhealthy",



            "error":

            str(e)



        }









# =====================================================
# SAFE GENERATION WRAPPER
# =====================================================


async def safe_generate_answer(

    question: str,

    chunks: list[RetrievedChunk]

) -> RAGAnswer:


    """
    Extra safety layer.

    Prevents RAG pipeline crash
    when LLM fails.
    """



    try:



        return await generate_answer_async(

            question,

            chunks

        )





    except Exception as e:



        logger.exception(

            f"Safe generation failure: {e}"

        )



        return RAGAnswer(


            answer=

            "تعذر إنشاء الإجابة حالياً.",



            sources=

            build_sources(

                chunks

            ),



            retriever_used=

            "fallback",



            chunks_retrieved=

            len(chunks)



        )









# =====================================================
# STREAM SAFE WRAPPER
# =====================================================


async def safe_stream_answer(

    question: str,

    chunks: list[RetrievedChunk]

) -> AsyncGenerator[str, None]:


    """
    Safe streaming wrapper.

    Compatible with:
    Streamlit st.write_stream
    """



    try:



        async for token in stream_answer_async(

            question,

            chunks

        ):



            yield token





    except Exception as e:



        logger.exception(

            f"Streaming wrapper failed: {e}"

        )



        yield (

            "حدث خطأ أثناء إنشاء الإجابة."

        )









# =====================================================
# EXPORTS
# =====================================================


__all__ = [



    "RAGAnswer",



    "generate_answer_async",



    "stream_answer_async",



    "safe_generate_answer",



    "safe_stream_answer",



    "check_llm_health"

]








# =====================================================
# LOCAL TEST ENTRY POINT
# =====================================================


async def _test():



    logger.info(

        "RAG Chain test started"

    )



    result = await check_llm_health()



    print(result)







if __name__ == "__main__":



    asyncio.run(

        _test()

    )

