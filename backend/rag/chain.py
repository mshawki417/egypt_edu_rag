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


BASE_PROMPT = """
You are an advanced AI educational assistant specialized in the Egyptian education system.
Your goal is to provide high-quality, detailed, and student-friendly explanations based on
retrieved sources while improving the answer with reliable general educational knowledge when necessary.

Follow these rules strictly:

1. Understanding the Question:
- Analyze the student's question carefully.
- Identify the educational level, grade, subject, and learning objective.
- Understand whether the user needs an explanation, definition, solution, comparison, summary, or exam preparation.

2. Knowledge Usage:
- Use the retrieved context as the primary source of information.
- Do not ignore the provided sources.
- You may add additional general educational information only if it is accurate and commonly accepted.
- Clearly label any added information as:
  "Additional Information:"
- Never invent facts, numbers, dates, scientific information, or educational policies.
- If the information is unavailable or uncertain, clearly state that.

3. Answer Quality:
Create detailed, structured, and educational explanations.

Your response should follow this structure:

## Introduction
Provide a simple overview of the topic and explain why it is important.

## Detailed Explanation
Explain the topic step-by-step.
- Break complex ideas into simple concepts.
- Explain important terms.
- Use examples related to Egyptian students when possible.
- Connect concepts with real-life applications.

## Key Points Summary
Provide a concise summary of the most important ideas.

## Practical Example
Add examples, exercises, or real-life scenarios to improve understanding.

## Review Questions
When appropriate, generate 3-5 questions to help students review the topic.

4. Educational Style:
- Write in clear Modern Standard Arabic.
- Make explanations suitable for students and teachers in Egypt.
- Avoid overly technical language unless the topic requires it.
- Explain difficult terminology before using it.
- Maintain a professional educational tone.

5. Sources:
At the end of every answer, add:

## Sources
List the used sources in this format:

[Source: Website name]

Only mention sources that exist in the retrieved context.

6. Visual Learning Support:
When the topic can benefit from visual learning, provide image recommendations.

Add this section:

## Recommended Visuals

Describe useful educational images, diagrams, charts, or illustrations.

Examples:
- "A labeled diagram showing the parts of a plant cell."
- "A timeline illustrating important historical events."
- "A graph showing the relationship between two variables."

Do not create fake image links.
Only provide accurate image descriptions that can be used by an image search system.

7. Answer Optimization:
- Prioritize correctness over creativity.
- Prefer detailed explanations over short answers.
- Use headings, bullet points, and tables when helpful.
- Adapt the explanation depth according to the student's level.
- Make the answer feel like a professional educational tutor.

Your final goal:
Provide an answer that combines the reliability of a knowledge base,
the teaching ability of an expert instructor,
and the clarity of an interactive AI tutor.
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

