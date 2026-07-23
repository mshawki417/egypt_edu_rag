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

from duckduckgo_search import DDGS



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

SCRAPER_CACHE = TTLCache(maxsize=1000, ttl=3600)


REQUEST_LIMIT = asyncio.Semaphore(5)




# ==========================
# Fetch URL
# ==========================

async def fetch_url(

    client,

    url

):


    if url in SCRAPER_CACHE:

        return SCRAPER_CACHE[url]



    async with REQUEST_LIMIT:

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


                doc = fitz.open(stream=response.content, filetype="pdf")
                data = ""
                for page in doc:
                    data += page.get_text()
                doc.close()



            else:


                soup = BeautifulSoup(

                    response.text,

                    "html.parser"

                )


                data = soup.get_text(

                    separator=" ",

                    strip=True

                )



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


    logger.info(f"DDG Search for: {query}")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            return [res['href'] for res in results if 'href' in res]
    except Exception as e:
        logger.error(f"DDGS error: {e}")
        return []




# ==========================
# Main Scraper
# ==========================

async def async_scrape_for_query(meta):


    query = meta.search_query



    logger.info(

        f"Searching: {query}"

    )



    urls = build_search_urls(

        query

    )



    documents=[]



    async with httpx.AsyncClient(

        headers={

            "User-Agent":
            "Mozilla/5.0"

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

                        "query":query

                    }

                )

            )



    logger.info(

        f"Collected {len(documents)} documents"

    )


    return documents
