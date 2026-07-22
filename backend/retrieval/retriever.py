"""
backend/retrieval/retriever.py
BM25, Dense (FAISS), and Hybrid retriever implementations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

import numpy as np
from loguru import logger

from backend.preprocessing.chunker import Chunk
from config.settings import retrieval_cfg

RetrieverType = Literal["bm25", "dense", "hybrid"]


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    retriever: str


# ── BM25 Retriever ─────────────────────────────────────────────────────────────
class BM25Retriever:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._bm25 = None

    def index(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi
        self._chunks = chunks
        tokenized = [c.text.split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25: indexed {len(chunks)} chunks")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(query.split())
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [RetrievedChunk(chunk=self._chunks[i], score=float(scores[i]), retriever="bm25")
                for i in top_idx if scores[i] > 0]


# ── Dense Retriever ────────────────────────────────────────────────────────────
class DenseRetriever:
    def __init__(self):
        self._chunks: list[Chunk] = []
        self._model = None
        self._index = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {retrieval_cfg.embedding_model}")
            self._model = SentenceTransformer(retrieval_cfg.embedding_model)

    def index(self, chunks: list[Chunk]):
        import faiss
        self._load_model()
        self._chunks = chunks
        texts = [c.text for c in chunks]
        logger.info(f"Dense: encoding {len(texts)} chunks…")
        embeddings = self._model.encode(texts, batch_size=32, show_progress_bar=True,
                                        convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings.astype("float32"))
        logger.info(f"Dense: FAISS index built (dim={dim})")

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if self._index is None:
            return []
        self._load_model()
        q_emb = self._model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self._index.search(q_emb.astype("float32"), top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievedChunk(chunk=self._chunks[idx],
                                          score=float(score), retriever="dense"))
        return results


# ── Hybrid Retriever ───────────────────────────────────────────────────────────
class HybridRetriever:
    def __init__(self):
        self._bm25   = BM25Retriever()
        self._dense  = DenseRetriever()
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]):
        self._chunks = chunks
        self._bm25.index(chunks)
        self._dense.index(chunks)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        bm25_results  = self._bm25.search(query, top_k * 2)
        dense_results = self._dense.search(query, top_k * 2)

        # Normalize + fuse scores
        scores: dict[str, float] = {}
        chunks_map: dict[str, Chunk] = {}

        w_bm25  = retrieval_cfg.bm25_weight
        w_dense = 1 - w_bm25

        max_bm25  = max((r.score for r in bm25_results),  default=1)
        max_dense = max((r.score for r in dense_results), default=1)

        for r in bm25_results:
            cid = r.chunk.chunk_id
            scores[cid]     = scores.get(cid, 0) + w_bm25 * (r.score / max(max_bm25, 1e-9))
            chunks_map[cid] = r.chunk

        for r in dense_results:
            cid = r.chunk.chunk_id
            scores[cid]     = scores.get(cid, 0) + w_dense * (r.score / max(max_dense, 1e-9))
            chunks_map[cid] = r.chunk

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [RetrievedChunk(chunk=chunks_map[cid], score=score, retriever="hybrid")
                for cid, score in ranked]


# ── Factory ────────────────────────────────────────────────────────────────────
def build_retriever(strategy: RetrieverType = "hybrid"):
    logger.info(f"Using retriever: {strategy}")
    if strategy == "bm25":
        return BM25Retriever()
    elif strategy == "dense":
        return DenseRetriever()
    return HybridRetriever()
