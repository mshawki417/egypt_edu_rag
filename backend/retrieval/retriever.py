"""
backend/retrieval/retriever.py

Three retrieval strategies:
  1. BM25Retriever       — keyword / TF-IDF (fast, no model needed)
  2. DenseRetriever      — semantic embeddings via sentence-transformers + FAISS
  3. HybridRetriever     — merges BM25 + Dense with Reciprocal Rank Fusion (RRF)

All retrievers work in-memory (ephemeral) — no persistent DB required.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config.settings import retrieval_cfg
from backend.preprocessing.chunker import Chunk
from backend.preprocessing.cleaner import normalize_arabic


# ── Result dataclass ───────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    retriever: str


# ── Tokenizer (Arabic-aware) ───────────────────────────────────────────────────
def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer after Arabic normalization."""
    return normalize_arabic(text).split()


# ── Base class ─────────────────────────────────────────────────────────────────
class BaseRetriever(ABC):
    @abstractmethod
    def index(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[RetrievedChunk]: ...


# ── 1. BM25 ───────────────────────────────────────────────────────────────────
class BM25Retriever(BaseRetriever):
    """Keyword-based retrieval using BM25Okapi."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        tokenized = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25: indexed {len(chunks)} chunks")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._bm25:
            return []
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(self._chunks[i], float(scores[i]), "bm25")
            for i in top_indices
            if scores[i] > 0
        ]


# ── 2. Dense (FAISS + SentenceTransformers) ───────────────────────────────────
class DenseRetriever(BaseRetriever):
    """
    Semantic dense retrieval.
    Uses a multilingual sentence-transformers model + FAISS flat index.
    """

    def __init__(self, model_name: str | None = None) -> None:
        model_name = model_name or retrieval_cfg.embedding_model
        logger.info(f"Loading embedding model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._chunks: list[Chunk] = []
        self._index = None   # faiss.IndexFlatIP

    def index(self, chunks: list[Chunk]) -> None:
        import faiss  # lazy import

        self._chunks = chunks
        texts = [c.text for c in chunks]

        logger.info(f"Dense: encoding {len(texts)} chunks…")
        embeddings = self._model.encode(
            texts, batch_size=32, show_progress_bar=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)   # inner product = cosine (normalized)
        self._index.add(embeddings)
        logger.info(f"Dense: FAISS index built (dim={dim})")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._index is None:
            return []
        q_emb = self._model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")
        scores, indices = self._index.search(q_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                results.append(RetrievedChunk(self._chunks[idx], float(score), "dense"))
        return results

    def reset(self) -> None:
        """Free FAISS memory (call after session ends)."""
        self._index = None
        self._chunks = []


# ── 3. Hybrid (RRF) ───────────────────────────────────────────────────────────
class HybridRetriever(BaseRetriever):
    """
    Combines BM25 + Dense using Reciprocal Rank Fusion.

    RRF score: sum of 1 / (k + rank_i) for each retriever.
    k=60 is the standard value from the original RRF paper.
    """

    def __init__(self, bm25_weight: float | None = None) -> None:
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.bm25_weight = bm25_weight or retrieval_cfg.bm25_weight
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        k_rrf = 60
        n_candidates = top_k * 3   # fetch more, then fuse

        bm25_results = self.bm25.search(query, n_candidates)
        dense_results = self.dense.search(query, n_candidates)

        # Map chunk_id → score
        rrf_scores: dict[str, float] = {}
        id_to_chunk: dict[str, Chunk] = {}

        for rank, res in enumerate(bm25_results):
            cid = res.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + self.bm25_weight / (k_rrf + rank + 1)
            id_to_chunk[cid] = res.chunk

        dense_w = 1.0 - self.bm25_weight
        for rank, res in enumerate(dense_results):
            cid = res.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + dense_w / (k_rrf + rank + 1)
            id_to_chunk[cid] = res.chunk

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]
        return [
            RetrievedChunk(id_to_chunk[cid], rrf_scores[cid], "hybrid")
            for cid in sorted_ids
        ]


# ── Factory ───────────────────────────────────────────────────────────────────
RetrieverType = Literal["bm25", "dense", "hybrid"]


def build_retriever(strategy: RetrieverType = "hybrid") -> BaseRetriever:
    """Instantiate the chosen retriever strategy."""
    mapping: dict[str, type[BaseRetriever]] = {
        "bm25": BM25Retriever,
        "dense": DenseRetriever,
        "hybrid": HybridRetriever,
    }
    cls = mapping.get(strategy, HybridRetriever)
    logger.info(f"Using retriever: {strategy}")
    return cls()
