"""
High Performance Arabic Chunker
Optimized for Real-Time RAG
"""


from __future__ import annotations


import re

from dataclasses import dataclass, field

from loguru import logger


from config.settings import retrieval_cfg

from backend.scraper.live_scraper import RawDocument

from backend.preprocessing.cleaner import (
    clean_text,
    is_arabic_heavy,
    extract_arabic_sections
)




# ==========================
# Schema
# ==========================


@dataclass
class Chunk:

    text:str

    chunk_id:str

    doc_id:str

    source_url:str

    title:str


    metadata:dict = field(
        default_factory=dict
    )





# ==========================
# Sentence Splitter
# ==========================


def split_sentences(
    text:str
):


    sentences = re.split(

        r'(?<=[.!؟])\s+',

        text

    )


    return [

        s.strip()

        for s in sentences

        if len(s.strip())>5

    ]





# ==========================
# Smart Chunking
# ==========================


def create_chunks(

    sentences,

    max_chars,

    overlap

):


    chunks=[]

    current=[]

    length=0



    for sentence in sentences:


        sentence_len=len(sentence)



        if length + sentence_len <= max_chars:


            current.append(sentence)

            length += sentence_len



        else:


            if current:

                chunks.append(
                    " ".join(current)
                )


            # overlap sentences

            overlap_text=[]

            overlap_len=0


            for s in reversed(current):


                if overlap_len+len(s)<=overlap:

                    overlap_text.insert(
                        0,
                        s
                    )

                    overlap_len+=len(s)

                else:

                    break



            current=overlap_text+[sentence]


            length=sum(
                len(x)
                for x in current
            )



    if current:

        chunks.append(
            " ".join(current)
        )


    return chunks





# ==========================
# Process Document
# ==========================


def chunk_document(
    doc:RawDocument
):


    cleaned=clean_text(
        doc.content
    )



    # Arabic filtering

    if not is_arabic_heavy(
        cleaned,
        threshold=0.15
    ):


        arabic_text=extract_arabic_sections(
            cleaned
        )


        if len(arabic_text)>100:

            cleaned=arabic_text


        else:

            logger.debug(
                f"Skipped non Arabic {doc.source_url}"
            )

            return []




    sentences=split_sentences(
        cleaned
    )



    raw_chunks=create_chunks(

        sentences,

        retrieval_cfg.chunk_size,

        retrieval_cfg.chunk_overlap

    )



    chunks=[]


    total=len(raw_chunks)



    for idx,text in enumerate(raw_chunks):


        chunks.append(

            Chunk(

                text=text,

                chunk_id=f"{doc.doc_id}_{idx}",

                doc_id=doc.doc_id,

                source_url=doc.source_url,

                title=doc.title,


                metadata={

                    **doc.metadata,


                    "chunk_index":idx,


                    "chunk_total":total,


                    "doc_type":doc.doc_type,


                    "characters":len(text)

                }

            )

        )



    return chunks





# ==========================
# Batch Processing
# ==========================


def process_documents(
    docs:list[RawDocument]
):


    all_chunks=[]



    for doc in docs:


        chunks=chunk_document(
            doc
        )


        all_chunks.extend(
            chunks
        )


    logger.info(

        f"""

        Documents:
        {len(docs)}

        Chunks:
        {len(all_chunks)}

        """

    )


    return all_chunks
