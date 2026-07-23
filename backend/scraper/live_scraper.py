"""
Live Scraper
Real-Time RAG Source Collector
"""

from __future__ import annotations


import asyncio
import hashlib

import httpx

from bs4 import BeautifulSoup
from dataclasses import dataclass, field

from loguru import logger
from cachetools import TTLCache

import fitz

from ddgs import DDGS



# ==========================
# Document Schema
# ==========================

@dataclass
class RawDocument:

    content: str

    source_url: str

    title: str = ""

    doc_id: str = ""

    doc_type: str = "web"

    metadata: dict = field(
        default_factory=dict
    )



# ==========================
# Cache
# ==========================

SCRAPER_CACHE = TTLCache(
    maxsize=1000,
    ttl=3600
)



# ==========================
# Async Request Limiter
# ==========================

def get_request_limit():

    """
    Creates semaphore per current asyncio task/event loop.
    Prevents Streamlit asyncio.run() loop conflicts.
    """

    task = asyncio.current_task()


    if task is None:

        return asyncio.Semaphore(5)



    if not hasattr(
        task,
        "_rag_request_limit"
    ):

        task._rag_request_limit = asyncio.Semaphore(
            5
        )


    return task._rag_request_limit





# ==========================
# Domain Rules
# ==========================


ALLOWED_DOMAINS = [

    "moe.gov.eg",

    "ekb.eg",

    "study.ekb.eg",

    "youm7.com",

    "elwatannews.com",

    "marefa.org",

    "wikipedia.org"

]



BLOCKED_DOMAINS = [

    "reddit.com",

    "facebook.com",

    "instagram.com",

    "pinterest.com",

    "tiktok.com"

]





def valid_url(url: str):


    if any(
        domain in url
        for domain in BLOCKED_DOMAINS
    ):

        return False



    if ALLOWED_DOMAINS:

        return any(
            domain in url
            for domain in ALLOWED_DOMAINS
        )



    return True





# ==========================
# Fetch URL
# ==========================


async def fetch_url(

    client,

    url

):


    if url in SCRAPER_CACHE:

        return SCRAPER_CACHE[url]



    async with get_request_limit():

        try:


            response = await client.get(

                url,

                timeout=20

            )



            response.raise_for_status()



            content_type = response.headers.get(

                "content-type",

                ""

            )



            if "pdf" in content_type:


                pdf = fitz.open(

                    stream=response.content,

                    filetype="pdf"

                )


                data = ""


                for page in pdf:

                    data += page.get_text()



                pdf.close()



            else:


                soup = BeautifulSoup(

                    response.text,

                    "html.parser"

                )



                for tag in soup(

                    [

                        "script",

                        "style",

                        "nav",

                        "footer"

                    ]

                ):

                    tag.decompose()



                data = soup.get_text(

                    separator=" ",

                    strip=True

                )





            if len(data) < 200:

                return None




            SCRAPER_CACHE[url] = data



            return data





        except Exception as e:


            logger.warning(

                f"Fetch failed {url}: {e}"

            )


            return None






# ==========================
# Search URLs
# ==========================


def build_search_urls(query):


    logger.info(

        f"DDG Search for: {query}"

    )


    try:


        search_query = (

            f"{query} "

            "مصر منهج وزارة التربية والتعليم"

        )



        urls = []



        with DDGS() as ddgs:


            results = ddgs.text(

                search_query,

                region="eg-ar",

                safesearch="moderate",

                max_results=10

            )



            for result in results:


                url = result.get(

                    "href"

                )


                if url and valid_url(url):

                    urls.append(url)





        logger.info(

            f"Found {len(urls)} URLs"

        )



        return urls





    except Exception as e:


        logger.error(

            f"DDGS error: {e}"

        )


        return []







# ==========================
# Main Scraper
# ==========================


async def async_scrape_for_query(meta):


    query = meta.search_query



    logger.info(

        f"Searching: {query}"

    )



    urls = await asyncio.to_thread(

        build_search_urls,

        query

    )



    documents = []




    async with httpx.AsyncClient(

        headers={

            "User-Agent":

            (

                "Mozilla/5.0 "

                "(Windows NT 10.0; Win64; x64)"

            )

        },

        follow_redirects=True

    ) as client:




        results = await asyncio.gather(

            *[

                fetch_url(

                    client,

                    url

                )

                for url in urls

            ]

        )




        for url, content in zip(

            urls,

            results

        ):



            if not content:

                continue




            doc_id = hashlib.md5(

                url.encode()

            ).hexdigest()




            documents.append(

                RawDocument(

                    content=content,

                    source_url=url,

                    title=query,

                    doc_id=doc_id,

                    metadata={

                        "query": query

                    }

                )

            )





    logger.info(

        f"Collected {len(documents)} documents"

    )



    return documents
