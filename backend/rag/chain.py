"""
High Performance Async RAG Chain
OpenRouter + Context Optimization
"""


from __future__ import annotations


import json

from dataclasses import dataclass


import httpx

from loguru import logger


from config.settings import llm_cfg

from backend.retrieval.retriever import RetrievedChunk




OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)



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


    answer:str


    sources:list[dict]


    retriever_used:str


    chunks_retrieved:int





# =========================
# Context
# =========================


def build_context(

    chunks:list[RetrievedChunk],

    max_chunks=5

):


    selected = chunks[:max_chunks]


    parts=[]



    for idx, item in enumerate(

        selected,

        1

    ):


        chunk=item.chunk



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

        chunks

    )


    answer = ""



    try:


        async with httpx.AsyncClient(

            timeout=60

        ) as client:


            response = await client.post(

                OPENROUTER_URL,

                headers=headers(),

                json=payload(

                    question,

                    context

                )

            )



            response.raise_for_status()



            data=response.json()



            answer=(

                data

                ["choices"]

                [0]

                ["message"]

                ["content"]

            )



    except Exception as e:


        logger.exception(

            e

        )


        answer=(

            "حدث خطأ أثناء توليد الإجابة."

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


            async for line in response.aiter_lines():


                if not line:

                    continue



                if line.startswith(

                    "data:"

                ):


                    data=line[5:].strip()



                    if data=="[DONE]":

                        break



                    try:


                        obj=json.loads(

                            data

                        )


                        token=(

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
