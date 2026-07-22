"""
Production Hybrid Retriever
FAISS + BM25 + Persistent Cache
"""

from __future__ import annotations


import re

from dataclasses import dataclass

import numpy as np

from loguru import logger


from backend.preprocessing.chunker import Chunk

from config.settings import retrieval_cfg



# =============================
# Types
# =============================

RetrieverType = str



@dataclass
class RetrievedChunk:

    chunk: Chunk

    score: float

    retriever: str



# =============================
# Embedding Model Cache
# =============================

_embedding_model = None



def get_embedding_model():

    global _embedding_model


    if _embedding_model is None:

        from sentence_transformers import SentenceTransformer


        logger.info(
            "Loading embedding model..."
        )


        _embedding_model = SentenceTransformer(

            retrieval_cfg.embedding_model

        )


    return _embedding_model




# =============================
# Tokenizer
# =============================

def tokenize(text):

    text = text.lower()


    text = re.sub(

        r"[^\w\s]",

        " ",

        text

    )


    return text.split()




# =============================
# BM25 Retriever
# =============================

class BM25Retriever:


    def __init__(self):

        self.chunks = []

        self.model = None




    def index(self, chunks):

        from rank_bm25 import BM25Okapi


        self.chunks = chunks


        corpus = [

            tokenize(
                c.text
            )

            for c in chunks

        ]


        self.model = BM25Okapi(
            corpus
        )


        logger.info(

            f"BM25 indexed {len(chunks)} chunks"

        )




    def search(self, query, top_k):


        if self.model is None:

            return []



        scores = self.model.get_scores(

            tokenize(query)

        )



        ids = np.argsort(

            scores

        )[::-1][:top_k]



        results = []


        for i in ids:


            if scores[i] <= 0:

                continue



            results.append(

                RetrievedChunk(

                    chunk=self.chunks[i],

                    score=float(
                        scores[i]
                    ),

                    retriever="bm25"

                )

            )


        return results




# =============================
# Dense FAISS Retriever
# =============================

class DenseRetriever:



    def __init__(self):

        self.chunks = []

        self.faiss_index = None

        self.model = get_embedding_model()




    # compatibility with orchestrator

    def index(self, chunks):

        self.index_chunks(
            chunks
        )




    def index_chunks(self, chunks):

        import faiss


        self.chunks = chunks



        texts = [

            c.text

            for c in chunks

        ]



        embeddings = self.model.encode(

            texts,

            batch_size=32,

            normalize_embeddings=True,

            show_progress_bar=False

        )



        embeddings = embeddings.astype(

            "float32"

        )



        dimension = embeddings.shape[1]



        self.faiss_index = faiss.IndexFlatIP(

            dimension

        )



        self.faiss_index.add(

            embeddings

        )



        logger.info(

            f"FAISS indexed {len(chunks)} chunks"

        )




    def search(self, query, top_k):


        if self.faiss_index is None:

            return []



        embedding = self.model.encode(

            [query],

            normalize_embeddings=True

        )



        embedding = embedding.astype(

            "float32"

        )



        scores, ids = self.faiss_index.search(

            embedding,

            top_k

        )



        results = []



        for score, idx in zip(

            scores[0],

            ids[0]

        ):


            if idx < 0:

                continue



            results.append(

                RetrievedChunk(

                    chunk=self.chunks[idx],

                    score=float(score),

                    retriever="dense"

                )

            )



        return results




# =============================
# Hybrid Retriever
# =============================

class HybridRetriever:



    def __init__(self):

        self.bm25 = BM25Retriever()

        self.dense = DenseRetriever()




    def index(self, chunks):


        self.bm25.index(

            chunks

        )


        self.dense.index(

            chunks

        )




    def search(self, query, top_k):


        bm_results = self.bm25.search(

            query,

            top_k * 3

        )


        dense_results = self.dense.search(

            query,

            top_k * 3

        )



        scores = {}

        mapping = {}




        for item in bm_results:


            cid = item.chunk.chunk_id


            scores[cid] = scores.get(

                cid,

                0

            ) + (

                retrieval_cfg.bm25_weight

                *

                item.score

            )


            mapping[cid] = item.chunk




        for item in dense_results:


            cid = item.chunk.chunk_id


            scores[cid] = scores.get(

                cid,

                0

            ) + (

                (1 - retrieval_cfg.bm25_weight)

                *

                item.score

            )


            mapping[cid] = item.chunk




        ranked = sorted(

            scores.items(),

            key=lambda x: x[1],

            reverse=True

        )[:top_k]




        return [

            RetrievedChunk(

                chunk=mapping[cid],

                score=float(score),

                retriever="hybrid"

            )

            for cid, score in ranked

        ]




# =============================
# Factory
# =============================

def build_retriever(strategy="hybrid"):


    if strategy == "bm25":

        return BM25Retriever()



    if strategy == "dense":

        return DenseRetriever()



    return HybridRetriever()
