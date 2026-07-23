"""
Live Scraper
Real-Time RAG Source Collector
"""

from __future__ import annotations


import asyncio
import hashlib

from weakref import WeakKeyDictionary

import httpx

import fitz

from bs4 import BeautifulSoup

from dataclasses import dataclass, field

from cachetools import TTLCache

from loguru import logger

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
# Async Semaphore
# ==========================


_semaphores: "WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = WeakKeyDictionary()



def get_request_limit():


    loop = asyncio.get_running_loop()



    semaphore = _semaphores.get(loop)



    if semaphore is None:


        semaphore = asyncio.Semaphore(5)



        _semaphores[loop] = semaphore



    return semaphore







# ==========================
# Domains
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



    return any(

        domain in url

        for domain in ALLOWED_DOMAINS

    )







# ==========================
# Fetch
# ==========================


async def fetch_url(

    client,

    url

):


    if url in SCRAPER_CACHE:


        return SCRAPER_CACHE[url]




    semaphore = get_request_limit()



    async with semaphore:


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




            if "pdf" in content_type.lower():


                pdf = fitz.open(

                    stream=response.content,

                    filetype="pdf"

                )



                text = ""



                for page in pdf:
