"""
High Performance Async RAG Chain
OpenRouter + Context Optimization
"""


from __future__ import annotations


import asyncio

import json

from dataclasses import dataclass


import httpx

from loguru import logger


from config.settings import llm_cfg

from backend.retrieval.retriever import RetrievedChunk




OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)



MAX_RETRIES = 3





# =========================
# Prompts
# =========================


BASE_PROMPT = """

أنت مساعد ذكاء اصطناعي متخصص في التعليم المصري.

القواعد:

- استخدم فقط المعلومات الموجودة في السياق.
- لا تخترع معلومات.
- إذا لم تجد الإجابة قل:
  لا توجد معلومات كافية.
- أجب باللغة العربية الفصحى البسيطة.
- استخدم نقاط عند الحاجة.

"""





# =========================
# Schema
# =========================


@dataclass
class RAGAnswer:


    answer: str


    sources: list[dict]


    retriever_used: str


    chunks_retrieved: int





# =========================
# Context
# =========================


def build_context(

    chunks: list[RetrievedChunk],

    max_chunks=5

):


    selected = chunks[:max_chunks]


    parts = []



    for idx, item in enumerate(

        selected,

        1

    ):


        chunk = item.chunk



        parts.append(

            f"""

[{idx}]

المصدر:
{chunk.source_url}


العنوان:
{chunk.title}


المحتوى:

{chunk.text}

"""

        )


    return "\n------------\n".join(parts)







# =========================
# Headers
# =========================


def headers():


    if not llm_cfg.openrouter_api_key:


        raise ValueError(

            "OPENROUTER_API_KEY is missing"

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







# =========================
# Payload
# =========================


def payload(

    question,

    context,

    stream=False

):


    return {


        "model":

        llm_cfg.model,



        "temperature":

        llm_cfg.temperature,



        "max_tokens":

        llm_cfg.max_tokens,



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


السياق:

{context}


الإجابة:

"""

            }

        ]

    }





# =========================
# OpenRouter Request
# =========================


async def call_openrouter(

    client,

    question,

    context

):


    request_payload = payload(

        question,

        context

    )



    for attempt in range(

        MAX_RETRIES

    ):



        response = await client.post(

            OPENROUTER_URL,

            headers=headers(),

            json=request_payload

        )



        if response.status_code != 429:


            response.raise_for_status()

            return response.json()




        retry_after = response.headers.get(

            "retry-after"

        )



        if retry_after:


            wait_time = int(

                retry_after

            )


        else:


            wait_time = (

                5 *

                (attempt + 1)

            )



        logger.warning(

            f"OpenRouter rate limit 429. "
            f"Retrying after {wait_time}s "
            f"(attempt {attempt + 1}/{MAX_RETRIES})"

        )



        await asyncio.sleep(

            wait_time

        )



    raise RuntimeError(

        "OpenRouter unavailable after retries"

    )







# =========================
# Generate
# =========================


async def generate_answer_async(

    question,

    chunks

):


    if not chunks:


        return RAGAnswer(

            answer="لا توجد معلومات كافية.",

            sources=[],

            retriever_used="none",

            chunks_retrieved=0

        )



    context = build_context(

        chunks,

        max_chunks=4

    )


    answer = ""



    try:


        async with httpx.AsyncClient(

            timeout=90

        ) as client:



            data = await call_openrouter(

                client,

                question,

                context

            )



            answer = (

                data

                ["choices"]

                [0]

                ["message"]

                ["content"]

            )



    except Exception as e:


        logger.exception(

            f"LLM generation failed: {e}"

        )


        answer = (

            "الخدمة غير متاحة حالياً، "
            "يرجى المحاولة مرة أخرى."

        )





    return RAGAnswer(


        answer=answer,


        sources=[

            {

                "title":

                c.chunk.title,


                "url":

                c.chunk.source_url,


                "score":

                round(

                    c.score,

                    4

                )

            }

            for c in chunks

        ],



        retriever_used=

        chunks[0].retriever,



        chunks_retrieved=

        len(chunks)

    )








# =========================
# Streaming
# =========================


async def stream_answer_async(

    question,

    chunks

):


    context = build_context(

        chunks

    )


    async with httpx.AsyncClient(

        timeout=120

    ) as client:


        async with client.stream(

            "POST",

            OPENROUTER_URL,

            headers=headers(),

            json=payload(

                question,

                context,

                True

            )

        ) as response:



            if response.status_code == 429:


                logger.warning(

                    "OpenRouter streaming rate limited"

                )


                yield (

                    "الخدمة مشغولة حالياً، "
                    "حاول مرة أخرى."

                )

                return




            response.raise_for_status()



            async for line in response.aiter_lines():


                if not line:

                    continue



                if line.startswith(

                    "data:"

                ):


                    data = line[5:].strip()



                    if data == "[DONE]":

                        break



                    try:


                        obj = json.loads(

                            data

                        )


                        token = (

                            obj

                            ["choices"]

                            [0]

                            ["delta"]

                            .get(

                                "content",

                                ""

                            )

                        )


                        if token:

                            yield token



                    except Exception:

                        continue
