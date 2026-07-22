"""
backend/rag/chain.py
RAG chain: chunks + question → OpenRouter LLM → Arabic answer.
"""
from __future__ import annotations
import json
from dataclasses import dataclass

import httpx
from loguru import logger

from config.settings import llm_cfg
from backend.retrieval.retriever import RetrievedChunk

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """أنت مساعد تعليمي متخصص في المناهج الدراسية المصرية.
مهمتك الإجابة على أسئلة الطلاب والمعلمين بناءً على المصادر الرسمية.

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
        parts.append(f"[{i}] المصدر: {source}\nالعنوان: {rc.chunk.title or 'غير محدد'}\n{rc.chunk.text}")
    return "\n---\n".join(parts)


def _build_headers() -> dict:
    key = llm_cfg.openrouter_api_key
    if not key:
        raise ValueError("OPENROUTER_API_KEY غير موجود — أضفه في Streamlit Secrets")
    return {
        "Authorization": f"Bearer {key}",
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
            {"role": "user", "content": USER_TEMPLATE.format(question=question, context=context)},
        ],
    }


def _no_chunks_answer() -> RAGAnswer:
    return RAGAnswer(
        answer="لم أتمكن من العثور على معلومات كافية للإجابة. يرجى إعادة صياغة السؤال.",
        sources=[], retriever_used="none", chunks_retrieved=0,
    )


def generate_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> RAGAnswer:
    if not retrieved_chunks:
        return _no_chunks_answer()
    context = _format_context(retrieved_chunks)
    logger.info(f"Calling OpenRouter ({llm_cfg.model}) — {len(retrieved_chunks)} chunks")
    try:
        with httpx.Client(timeout=60) as client:
            resp = client.post(OPENROUTER_URL, headers=_build_headers(),
                               json=_build_payload(question, context, stream=False))
            resp.raise_for_status()
            data = resp.json()
        # Handle error object in response
        if "error" in data:
            err_msg = data["error"].get("message", "unknown error")
            logger.error(f"OpenRouter error: {err_msg}")
            return RAGAnswer(answer=f"⚠️ خطأ من نموذج الذكاء الاصطناعي: {err_msg}",
                             sources=[], retriever_used="error", chunks_retrieved=0)
        answer_text = data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.error(f"LLM call failed: {exc}")
        return RAGAnswer(answer=f"⚠️ تعذر الاتصال بنموذج الذكاء الاصطناعي: {exc}",
                         sources=[], retriever_used="error", chunks_retrieved=0)
    sources = [{"url": rc.chunk.source_url, "title": rc.chunk.title or rc.chunk.source_url,
                "score": round(rc.score, 4), "retriever": rc.retriever}
               for rc in retrieved_chunks]
    return RAGAnswer(answer=answer_text, sources=sources,
                     retriever_used=retrieved_chunks[0].retriever,
                     chunks_retrieved=len(retrieved_chunks))


def stream_answer(question: str, retrieved_chunks: list[RetrievedChunk]):
    """Generator yielding text tokens for Streamlit streaming."""
    if not retrieved_chunks:
        yield "لم أتمكن من العثور على معلومات كافية."
        return
    context = _format_context(retrieved_chunks)
    logger.info(f"Streaming from OpenRouter ({llm_cfg.model})")
    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", OPENROUTER_URL, headers=_build_headers(),
                               json=_build_payload(question, context, stream=True)) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    line = line.strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            chunk_data = json.loads(line[6:])
                            # Handle error object
                            if "error" in chunk_data:
                                err = chunk_data["error"].get("message", "unknown")
                                yield f"\n⚠️ خطأ: {err}"
                                return
                            delta = chunk_data["choices"][0].get("delta", {})
                            token = delta.get("content", "")
                            if token:
                                yield token
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
    except Exception as exc:
        logger.error(f"Streaming failed: {exc}")
        yield f"\n⚠️ انقطع الاتصال بنموذج الذكاء الاصطناعي: {exc}"
