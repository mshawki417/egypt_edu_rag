"""
Production Hybrid Retriever
FAISS + BM25 + Persistent Cache
"""


from __future__ import annotations


import os
import pickle
import re

from dataclasses import dataclass

import numpy as np

from loguru import logger


from backend.preprocessing.chunker import Chunk

from config.settings import retrieval_cfg




RetrieverType = str





@dataclass
class RetrievedChunk:

    chunk:Chunk

    score:float

    retriever:str





# =============================
# Global Embedding Model Cache
# =============================


_embedding_model=None




def get_embedding_model():


    global _embedding_model


    if _embedding_model is None:


        from sentence_transformers import SentenceTransformer


        logger.info(
            "Loading embedding model..."
        )


        _embedding_model=SentenceTransformer(

            retrieval_cfg.embedding_model

        )


    return _embedding_model





# =============================
# Arabic Tokenizer
# =============================


def tokenize(text):


    text=text.lower()


    text=re.sub(
        r"[^\w\s]",
        " ",
        text
    )


    return text.split()





# =============================
# BM25
# =============================


class BM25Retriever:


    def __init__(self):

        self.chunks=[]

        self.model=None




    def index(self,chunks):


        from rank_bm25 import BM25Okapi


        self.chunks=chunks


        corpus=[

            tokenize(c.text)

            for c in chunks

        ]


        self.model=BM25Okapi(
            corpus
        )


        logger.info(
            f"BM25 indexed {len(chunks)}"
        )




    def search(self,query,top_k):


        if not self.model:

            return []



        scores=self.model.get_scores(

            tokenize(query)

        )



        ids=np.argsort(
            scores
        )[::-1][:top_k]



        return [

            RetrievedChunk(

                chunk=self.chunks[i],

                score=float(scores[i]),

                retriever="bm25"

            )

            for i in ids

            if scores[i]>0

        ]





# =============================
# Dense FAISS
# =============================


class DenseRetriever:


    def __init__(self):

        self.chunks=[]

        self.index=None

        self.model=get_embedding_model()




    def index_chunks(self,chunks):


        import faiss



        self.chunks=chunks


        texts=[

            c.text

            for c in chunks

        ]



        embeddings=self.model.encode(

            texts,

            batch_size=64,

            normalize_embeddings=True,

            show_progress_bar=False

        )



        dim=embeddings.shape[1]



        self.index=faiss.IndexFlatIP(
            dim
        )


        self.index.add(

            embeddings.astype(
                "float32"
            )

        )


        logger.info(
            "FAISS ready"
        )





    def search(self,query,top_k):


        if self.index is None:

            return []



        emb=self.model.encode(

            [query],

            normalize_embeddings=True

        )



        scores,ids=self.index.search(

            emb.astype(
                "float32"
            ),

            top_k

        )



        return [

            RetrievedChunk(

                chunk=self.chunks[i],

                score=float(score),

                retriever="dense"

            )


            for score,i in zip(

                scores[0],

                ids[0]

            )

            if i>=0

        ]





# =============================
# Hybrid
# =============================


class HybridRetriever:


    def __init__(self):

        self.bm25=BM25Retriever()

        self.dense=DenseRetriever()




    def index(self,chunks):


        self.bm25.index(
            chunks
        )


        self.dense.index_chunks(
            chunks
        )




    def search(self,query,top_k):


        bm=self.bm25.search(
            query,
            top_k*3
        )


        dense=self.dense.search(

            query,

            top_k*3

        )



        scores={}

        mapping={}



        for r in bm:


            cid=r.chunk.chunk_id


            scores[cid]=scores.get(cid,0)+(

                retrieval_cfg.bm25_weight

                *

                r.score

            )


            mapping[cid]=r.chunk





        for r in dense:


            cid=r.chunk.chunk_id


            scores[cid]=scores.get(cid,0)+(

                (1-retrieval_cfg.bm25_weight)

                *

                r.score

            )


            mapping[cid]=r.chunk





        ranked=sorted(

            scores.items(),

            key=lambda x:x[1],

            reverse=True

        )[:top_k]



        return [

            RetrievedChunk(

                chunk=mapping[cid],

                score=float(score),

                retriever="hybrid"

            )


            for cid,score in ranked

        ]






# =============================
# Factory
# =============================


def build_retriever(strategy="hybrid"):


    if strategy=="bm25":

        return BM25Retriever()


    if strategy=="dense":

        return DenseRetriever()



    return HybridRetriever()
