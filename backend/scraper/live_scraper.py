"""
Live Web Scraper
Real-Time RAG Data Collector
"""

from __future__ import annotations


import asyncio

import httpx

from bs4 import BeautifulSoup

from loguru import logger



# ==========================
# Cache
# ==========================

SCRAPER_CACHE = {}



# ==========================
# Request Control
# ==========================

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

                data = response.content


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
# Search Sources
# ==========================

def build_search_urls(query):


    from urllib.parse import quote


    encoded = quote(query)


    return [

        f"https://duckduckgo.com/html/?q={encoded}",

        f"https://www.google.com/search?q={encoded}"

    ]



# ==========================
# Main RAG Scraper
# ==========================

async def async_scrape_for_query(meta):


    query = meta.search_query


    logger.info(
        f"Live scraping query: {query}"
    )


    urls = build_search_urls(
        query
    )


    documents = []



    async with httpx.AsyncClient(

        headers={

            "User-Agent":
            "Mozilla/5.0"

        },

        follow_redirects=True

    ) as client:



        tasks = [

            fetch_url(
                client,
                url
            )

            for url in urls

        ]



        results = await asyncio.gather(
            *tasks
        )



        for url, content in zip(
            urls,
            results
        ):


            if content:


                documents.append(

                    {

                        "source": url,

                        "content": content

                    }

                )


    logger.info(
        f"Scraped documents: {len(documents)}"
    )


    return documents
