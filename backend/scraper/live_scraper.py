"""
Real-Time Async Live Scraper
Wikipedia API + Static Sources + DuckDuckGo

Optimized for Real-Time RAG
"""

from __future__ import annotations


import re
import hashlib
import asyncio

from dataclasses import dataclass, field
from typing import List
from urllib.parse import quote, urlparse


import httpx
import fitz

from bs4 import BeautifulSoup
from loguru import logger
from cachetools import TTLCache


from config.settings import scraper_cfg
from backend.scraper.query_analyzer import QueryMetadata



# ============================
# Cache
# ============================

SCRAPER_CACHE = TTLCache(
    maxsize=200,
    ttl=900
)



# ============================
# Document Schema
# ============================


@dataclass
class RawDocument:


    content:str

    source_url:str

    title:str=""


    doc_type:str="html"


    metadata:dict = field(
        default_factory=dict
    )


    @property
    def doc_id(self):

        return hashlib.md5(
            self.source_url.encode()
        ).hexdigest()[:12]





# ============================
# Async Client
# ============================


def create_client():

    return httpx.AsyncClient(

        headers={

            "User-Agent":
            scraper_cfg.user_agent,


            "Accept-Language":
            "ar,en;q=0.8"

        },


        timeout=
        httpx.Timeout(
            scraper_cfg.timeout
        ),


        follow_redirects=True

    )





# ============================
# Fetch
# ============================


async def fetch_url(
    client,
    url
):


    if url in SCRAPER_CACHE:

        return SCRAPER_CACHE[url]



    try:


        response = await client.get(
            url
        )


        response.raise_for_status()


        data = response.text


        SCRAPER_CACHE[url]=data


        return data



    except Exception as e:


        logger.warning(
            f"Fetch failed {url}: {e}"
        )


        return ""





# ============================
# HTML Extraction
# ============================


def extract_html(
    html,
    url
):


    soup = BeautifulSoup(
        html,
        "lxml"
    )


    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header"
        ]
    ):

        tag.decompose()



    title = (

        soup.title.text.strip()

        if soup.title

        else ""

    )



    main = (

        soup.find(
            "div",
            id="mw-content-text"
        )

        or soup.find("article")

        or soup.body

    )



    text = main.get_text(
        "\n",
        strip=True
    )



    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )



    return RawDocument(

        content=text,

        source_url=url,

        title=title

    )





# ============================
# Wikipedia Async
# ============================


async def wikipedia_search(
    query
):


    url=(

        "https://ar.wikipedia.org/w/api.php?"

        f"action=query&list=search"

        f"&srsearch={quote(query)}"

        "&srlimit=3"

        "&format=json"

    )


    async with create_client() as client:


        try:

            res = await client.get(
                url
            )


            data=res.json()


            results=data["query"]["search"]



            tasks=[]


            for item in results:


                title=item["title"]


                api=(

                "https://ar.wikipedia.org/w/api.php?"

                f"action=query"

                f"&titles={quote(title)}"

                "&prop=extracts"

                "&explaintext=true"

                "&exchars=5000"

                "&format=json"

                )


                tasks.append(
                    fetch_url(
                        client,
                        api
                    )
                )



            pages = await asyncio.gather(
                *tasks
            )



            docs=[]


            for page in pages:


                if not page:
                    continue


                try:

                    data=eval(page)

                except:

                    continue


                for p in data.get(
                    "query",
                    {}
                ).get(
                    "pages",
                    {}
                ).values():


                    text=p.get(
                        "extract",
                        ""
                    )


                    if len(text)>200:


                        docs.append(

                            RawDocument(

                                content=text,

                                source_url=
                                "Wikipedia",

                                title=
                                p.get(
                                    "title",
                                    ""
                                )

                            )

                        )


            return docs



        except Exception as e:


            logger.error(
                e
            )


            return []





# ============================
# Static Sources
# ============================


async def scrape_static(
    urls
):


    async with create_client() as client:


        tasks=[

            fetch_url(
                client,
                u
            )

            for u in urls

        ]


        htmls = await asyncio.gather(
            *tasks
        )


        docs=[]


        for url,html in zip(
            urls,
            htmls
        ):


            if html:

                doc=extract_html(
                    html,
                    url
                )


                if len(doc.content)>200:

                    docs.append(
                        doc
                    )


        return docs





# ============================
# Main Function
# ============================


async def async_scrape_for_query(
    meta:QueryMetadata
):


    logger.info(
        f"Realtime search: {meta.search_query}"
    )


    tasks=[]



    # Wikipedia

    tasks.append(

        wikipedia_search(
            meta.search_query
        )

    )



    # Static

    urls=scraper_cfg.sources.get(
        meta.source_category,
        []
    )


    tasks.append(

        scrape_static(
            urls[:5]
        )

    )



    results=await asyncio.gather(
        *tasks
    )



    documents=[]


    for batch in results:

        documents.extend(
            batch
        )



    # Add metadata


    for doc in documents:

        doc.metadata.update(

            {

            "grade":meta.grade,

            "subject":meta.subject,

            "term":meta.term

            }

        )



    logger.info(

        f"Realtime documents: {len(documents)}"

    )


    return documents





# Sync wrapper for Streamlit

def scrape_for_query(meta):

    return asyncio.run(
        async_scrape_for_query(meta)
    )
