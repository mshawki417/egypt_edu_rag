"""
backend/preprocessing/chunker.py

Splits cleaned text into overlapping chunks, preserving paragraph
boundaries where possible (better for Arabic educational text).
"""
from __future__ import annotations
from dataclasses import dataclass, field

from config.settings import retrieval_cfg
from backend.scraper.live_scraper import RawDocument
from backend.preprocessing.cleaner import clean_text, is_arabic_heavy


@dataclass
class Chunk:
    """A single processable chunk ready for embedding."""
    text: str
    chunk_id: str
    doc_id: str
    source_url: str
    title: str
    metadata: dict = field(default_factory=dict)


def _split_by_paragraphs(text: str, max_chars: int) -> list[str]:
    """Split on double-newlines first, then hard-split overlong paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_chars:
            current = (current + "\n\n" + para).strip() if current else para
        else:
            if current:
                chunks.append(current)
            # Para itself might be too long → hard split
            if len(para) > max_chars:
                for i in range(0, len(para), max_chars):
                    chunks.append(para[i: i + max_chars])
            else:
                current = para

    if current:
        chunks.append(current)
    return chunks


def chunk_document(doc: RawDocument) -> list[Chunk]:
    """
    Clean and chunk a RawDocument into Chunk objects.

    Uses paragraph-aware splitting with character-level overlap
    (re-appends the tail of the previous chunk to the next one).
    """
    size = retrieval_cfg.chunk_size
    overlap = retrieval_cfg.chunk_overlap

    cleaned = clean_text(doc.content)

    # Skip docs with very little Arabic content (e.g. navigation pages)
    if not is_arabic_heavy(cleaned, threshold=0.15):
        return []

    raw_chunks = _split_by_paragraphs(cleaned, max_chars=size)

    chunks: list[Chunk] = []
    prev_tail = ""

    for idx, text in enumerate(raw_chunks):
        # Prepend overlap from previous chunk
        full_text = (prev_tail + " " + text).strip() if prev_tail else text
        prev_tail = text[-overlap:] if len(text) > overlap else text

        chunk = Chunk(
            text=full_text,
            chunk_id=f"{doc.doc_id}-{idx:04d}",
            doc_id=doc.doc_id,
            source_url=doc.source_url,
            title=doc.title,
            metadata={
                **doc.metadata,
                "chunk_index": idx,
                "chunk_total": len(raw_chunks),
                "doc_type": doc.doc_type,
            },
        )
        chunks.append(chunk)

    return chunks


def process_documents(docs: list[RawDocument]) -> list[Chunk]:
    """Process a list of RawDocuments into a flat list of Chunks."""
    all_chunks: list[Chunk] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
    return all_chunks
