"""
backend/preprocessing/chunker.py
Splits cleaned text into overlapping chunks with Arabic fallback.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from loguru import logger

from config.settings import retrieval_cfg
from backend.scraper.live_scraper import RawDocument
from backend.preprocessing.cleaner import clean_text, is_arabic_heavy, extract_arabic_sections


@dataclass
class Chunk:
    text: str
    chunk_id: str
    doc_id: str
    source_url: str
    title: str
    metadata: dict = field(default_factory=dict)


def _split_by_paragraphs(text: str, max_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i: i + max_chars])
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: RawDocument) -> list[Chunk]:
    size    = retrieval_cfg.chunk_size
    overlap = retrieval_cfg.chunk_overlap
    cleaned = clean_text(doc.content)

    if not is_arabic_heavy(cleaned, threshold=0.15):
        arabic_only = extract_arabic_sections(cleaned)
        if len(arabic_only.strip()) >= 100:
            cleaned = arabic_only
        else:
            ratio = sum(1 for c in cleaned if "\u0600" <= c <= "\u06FF") / max(len(cleaned), 1)
            logger.debug(f"Skipped doc (low Arabic): {doc.source_url} | ratio={ratio:.2%}")
            return []

    raw_chunks = [c for c in _split_by_paragraphs(cleaned, max_chars=size) if len(c.strip()) >= 30]
    if not raw_chunks:
        return []

    chunks: list[Chunk] = []
    prev_tail = ""
    for idx, text in enumerate(raw_chunks):
        full_text = (prev_tail + " " + text).strip() if prev_tail else text
        prev_tail = text[-overlap:] if len(text) > overlap else text
        chunks.append(Chunk(
            text=full_text,
            chunk_id=f"{doc.doc_id}-{idx:04d}",
            doc_id=doc.doc_id,
            source_url=doc.source_url,
            title=doc.title,
            metadata={**doc.metadata, "chunk_index": idx, "chunk_total": len(raw_chunks), "doc_type": doc.doc_type},
        ))
    return chunks


def process_documents(docs: list[RawDocument]) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc)
        logger.debug(f"Doc: {doc.source_url} | len={len(doc.content)} | chunks={len(chunks)}")
        all_chunks.extend(chunks)
    logger.info(f"Total chunks: {len(all_chunks)} from {len(docs)} docs")
    return all_chunks
