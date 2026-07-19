"""
backend/rag/chain.py

RAG chain: retrieved chunks + question → OpenRouter LLM → cited Arabic answer.
Uses OpenRouter free tier (openai-compatible API).
Free models: meta-llama/llama-3.1-8b-instruct:free
             mistralai/mistral-7b-instruct:free
             google/gemma-2-9b-it:free
"""
from __future__ import annotations
from dataclasses import dataclass

import httpx
from loguru import logger

from config.settings import llm_cfg
from backend.retrieval.retriever import RetrievedChunk


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """أنت مساعد تعليمي متخصص في المناهج الدراسية المصرية.
مهمتك الإجابة على أسئلة الطلاب والمعلمين بناءً على المصادر الرسمية لوزارة التربية والتعليم.

قواعد الإجابة:
١. استخدم فقط المعلومات الواردة في السياق المرفق.
٢. اذكر مصدر كل معلومة بين قوسين مربعين [المصدر: اسم الموقع].
٣. إذا لم تجد الإجابة في السياق، قل ذلك بوضوح ولا تخترع معلومات.
٤. أجب باللغة العربية الفصحى البسيطة المناسبة للطلاب.
٥. نظّم إجابتك في نقاط واضحة إذا كانت تحتوي على عدة عناصر."""

USER_TEMPLATE = """السؤال: {question}

السياق المسترجع:
{context}

أجب على السؤال بناءً على السياق أعلاه فقط."""


@dataclass
class RAGAnswer:
    answer: str
    sources: list[dict]
    retriever_used: str
    chunks_retrieved: int


def _format_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, rc in enumerate(chunks, 1):
        source = rc.chunk.metadata.get("source", rc.chunk.source_url)
        parts.append(
            f"[{i}] المصدر: {source}\n"
            f"العنوان: {rc.chunk.title or 'غير محدد'}\n"
            f"{rc.chunk.text}\n"
        )
    return "\n---\n".join(parts)


def _build_headers() -> dict:
    return {
        "Authorization": f"Bearer {llm_cfg.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/egypt-edu-rag",
        "X-Title": "Egypt Education RAG",
    }


def _build_payload(question: str, context: str, stream: bool = False) -> dict:
    return {
        "model": llm_cfg.model,
        "max_tokens": llm_cfg.max_tokens,
        "temperature": llm_cfg.temperature,
        "stream": stream,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                question=question, context=context
            )},
        ],
    }


def generate_answer(
    question: str,
    retrieved_chunks: list[RetrievedChunk],
) -> RAGAnswer:
    """Batch (non-streaming) answer via OpenRouter."""
    if not retrieved_chunks:
        return RAGAnswer(
            answer=(
                "لم أتمكن من العثور على معلومات كافية للإجابة على سؤالك. "
                "يرجى إعادة صياغة السؤال أو التحقق من اتصالك بالإنترنت."
            ),
            sources=[],
            retriever_used="none",
            chunks_retrieved=0,
        )

    context = _format_context(retrieved_chunks)
    logger.info(f"Calling OpenRouter ({llm_cfg.model}) — {len(retrieved_chunks)} chunks")

    with httpx.Client(timeout=60) as client:
        resp = client.post(
            OPENROUTER_URL,
            headers=_build_headers(),
            json=_build_payload(question, context, stream=False),
        )
        resp.raise_for_status()
        data = resp.json()

    answer_text = data["choices"][0]["message"]["content"]
    sources = [
        {
            "url": rc.chunk.source_url,
            "title": rc.chunk.title or rc.chunk.source_url,
            "score": round(rc.score, 4),
            "retriever": rc.retriever,
        }
        for rc in retrieved_chunks
    ]
    return RAGAnswer(
        answer=answer_text,
        sources=sources,
        retriever_used=retrieved_chunks[0].retriever,
        chunks_retrieved=len(retrieved_chunks),
    )


def stream_answer(question: str, retrieved_chunks: list[RetrievedChunk]):
    """
    Generator — yields text tokens for Streamlit streaming display.
    Uses OpenRouter SSE streaming (openai-compatible).
    """
    if not retrieved_chunks:
        yield "لم أتمكن من العثور على معلومات كافية."
        return

    context = _format_context(retrieved_chunks)
    logger.info(f"Streaming from OpenRouter ({llm_cfg.model})")

    with httpx.Client(timeout=120) as client:
        with client.stream(
            "POST",
            OPENROUTER_URL,
            headers=_build_headers(),
            json=_build_payload(question, context, stream=True),
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                line = line.strip()
                if not line or line == "data: [DONE]":
                    continue
                if line.startswith("data: "):
                    import json
                    try:
                        chunk = json.loads(line[6:])
                        delta = chunk["choices"][0].get("delta", {})
                        token = delta.get("content", "")
                        if token:
                            yield token
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
