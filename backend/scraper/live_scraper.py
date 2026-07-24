"""
Production Live Scraper
Real-Time RAG Source Collector
Optimized for Education RAG
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time

from urllib.parse import urlparse, urlunparse

from collections import defaultdict

from weakref import WeakKeyDictionary

import httpx
import fitz

from bs4 import BeautifulSoup

from dataclasses import dataclass, field

from cachetools import TTLCache

from loguru import logger

from ddgs import DDGS

# =====================================================
# CONFIG
# =====================================================


REQUEST_TIMEOUT = int(
    os.getenv(
        "SCRAPER_TIMEOUT",
        20
    )
)


CACHE_TTL = int(
    os.getenv(
        "SCRAPER_CACHE_TTL",
        3600
    )
)


CACHE_SIZE = int(
    os.getenv(
        "SCRAPER_CACHE_SIZE",
        1000
    )
)


MAX_RETRIES = 3


MAX_CONCURRENT_REQUESTS = 5



USER_AGENT = (
    "EgyptEducationRAG/1.0 "
    "(Educational Research Bot)"
)

# =====================================================
# Metrics
# =====================================================


METRICS = {

    "documents_collected":0,

    "fetch_errors":0,

    "blocked_urls":0,

    "cache_hits":0

}

# =====================================================
# Document Schema
# =====================================================


@dataclass
class RawDocument:


    content:str


    source_url:str


    title:str=""


    doc_id:str=""


    doc_type:str="web"


    metadata:dict=field(
        default_factory=dict
    )


# =====================================================
# Cache
# =====================================================


SCRAPER_CACHE = TTLCache(

    maxsize=CACHE_SIZE,

    ttl=CACHE_TTL

)

# =====================================================
# Async Rate Limit
# =====================================================


_semaphores = WeakKeyDictionary()


_host_limits = defaultdict(
    lambda: asyncio.Semaphore(
        MAX_CONCURRENT_REQUESTS
    )
)


def get_request_limit(host):


    return _host_limits[host]

# =====================================================
# Domains
# =====================================================


ALLOWED_DOMAINS=[


    "moe.gov.eg",

    "ekb.eg",

    "study.ekb.eg",

    "youm7.com",

    "elwatannews.com",

    "marefa.org",

    "wikipedia.org"

]

BLOCKED_DOMAINS=[


    "facebook.com",

    "instagram.com",

    "tiktok.com",

    "reddit.com",

    "pinterest.com"

]

# =====================================================
# URL Utilities
# =====================================================


def normalize_url(url:str)->str:


    """
    Normalize URL before fetching
    """


    if not url:

        return ""



    url=url.strip()



    if not url.startswith(
        ("http://","https://")
    ):

        url="https://"+url



    parsed=urlparse(url)



    clean=urlunparse(

        (

            parsed.scheme.lower(),

            parsed.netloc.lower(),

            parsed.path,

            "",

            parsed.query,

            ""

        )

    )


    return clean



def is_allowed_domain(url:str)->bool:


    try:

        parsed=urlparse(
            normalize_url(url)
        )


        domain=parsed.netloc.replace(
            "www.",
            ""
        )


        for blocked in BLOCKED_DOMAINS:


            if domain.endswith(blocked):

                METRICS["blocked_urls"]+=1

                return False




        return any(

            domain.endswith(
                allowed
            )

            for allowed in ALLOWED_DOMAINS

        )


    except Exception:

        return False


# backward compatibility

def valid_url(url:str):

    return is_allowed_domain(url)


# =====================================================
# Robots Basic Check
# =====================================================


async def allowed_by_robots(
    client,
    url
):


    """
    Minimal robots protection
    """

    try:

        parsed=urlparse(url)

        robots=f"{parsed.scheme}://{parsed.netloc}/robots.txt"


        response=await client.get(
            robots,
            timeout=5
        )


        if response.status_code!=200:

            return True



        path=parsed.path.lower()


        for line in response.text.splitlines():


            if "disallow:" in line.lower():


                blocked=line.split(":")[1].strip()


                if blocked and path.startswith(blocked):

                    return False



        return True



    except Exception:

        return True


# =====================================================
# Text Extraction
# =====================================================


def extract_text_from_html(
    html:str
):


    soup=BeautifulSoup(
        html,
        "html.parser"
    )


    title=""


    if soup.title:

        title=soup.title.text.strip()



    description=""


    meta=soup.find(
        "meta",
        attrs={
            "name":"description"
        }
    )


    if meta:

        description=meta.get(
            "content",
            ""
        )



    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside"
        ]
    ):

        tag.decompose()



    body=soup.get_text(

        separator=" ",

        strip=True

    )


    content=(

        title+

        "\n"+

        description+

        "\n"+

        body

    )


    return content.strip()







def extract_pdf_text(
    data:bytes
):


    try:

        pdf=fitz.open(

            stream=data,

            filetype="pdf"

        )


        text="\n".join(

            page.get_text()

            for page in pdf

        )


        pdf.close()


        return text



    except Exception as e:

        logger.error(
            f"PDF error {e}"
        )

        return ""


# =====================================================
# Fetch With Retry
# =====================================================


async def fetch_url(
    client,
    url
):


    url=normalize_url(
        url
    )


    if not is_allowed_domain(url):

        return None



    cache_key=hashlib.md5(

        url.encode()

    ).hexdigest()



    if cache_key in SCRAPER_CACHE:


        METRICS["cache_hits"]+=1

        return SCRAPER_CACHE[cache_key]


    host=urlparse(url).netloc


    semaphore=get_request_limit(
        host
    )



    async with semaphore:


        for attempt in range(
            MAX_RETRIES
        ):


            try:


                start=time.time()



                if not await allowed_by_robots(
                    client,
                    url
                ):

                    logger.warning(
                        f"robots blocked {url}"
                    )

                    return None




                response=await client.get(

                    url,

                    timeout=REQUEST_TIMEOUT

                )



                response.raise_for_status()



                content_type=response.headers.get(

                    "content-type",

                    ""

                ).lower()



                if not (

                    "text/html" in content_type

                    or

                    "pdf" in content_type

                ):

                    return None




                if "pdf" in content_type:


                    text=extract_pdf_text(

                        response.content

                    )

                    doc_type="pdf"



                else:


                    text=extract_text_from_html(

                        response.text

                    )

                    doc_type="web"





                if len(text)<200:

                    return None


                SCRAPER_CACHE[cache_key]=text



                logger.info(

                    f"""

                    FETCH OK

                    URL={url}

                    TYPE={doc_type}

                    SIZE={len(text)}

                    TIME={time.time()-start:.2f}s

                    """

                )



                return text





            except (

                httpx.TimeoutException,

                httpx.NetworkError

            ) as e:


                wait=2**attempt


                logger.warning(

                    f"Retry {attempt+1}/{MAX_RETRIES} {url}"

                )


                await asyncio.sleep(
                    wait
                )



            except Exception as e:


                METRICS["fetch_errors"]+=1


                logger.error(

                    f"Fetch failed {url}: {str(e)[:100]}"

                )

                break

    return None

# =====================================================
# Search
# =====================================================


def build_search_urls(
    query
):


    urls=[]


    try:


        search_query=(

            f"{query} "

            "وزارة التربية والتعليم مصر"

        )


        with DDGS() as ddgs:


            results=ddgs.text(

                search_query,

                region="eg-ar",

                safesearch="moderate",

                max_results=15

            )


            for item in results:


                url=item.get(
                    "href"
                )


                if url and valid_url(url):

                    urls.append(

                        normalize_url(url)

                    )



        return list(
            set(urls)
        )



    except Exception as e:


        logger.error(
            f"Search error {e}"
        )


        return []

# =====================================================
# Main Pipeline
# =====================================================


async def async_scrape_for_query(meta):


    query=meta.search_query



    urls=await asyncio.to_thread(

        build_search_urls,

        query

    )



    if not urls:

        return []




    documents=[]



    async with httpx.AsyncClient(

        headers={

            "User-Agent":USER_AGENT

        },

        follow_redirects=True

    ) as client:



        results=await asyncio.gather(

            *[

                fetch_url(
                    client,
                    url
                )

                for url in urls

            ],

            return_exceptions=True

        )





        for url,content in zip(

            urls,

            results

        ):


            if not content or isinstance(
                content,
                Exception
            ):

                continue





            doc_id=hashlib.md5(

                (

                    url+

                    content[:200]

                ).encode()

            ).hexdigest()






            documents.append(

                RawDocument(

                    content=content,

                    source_url=url,

                    title=query,

                    doc_id=doc_id,

                    metadata={

                        "query":query,

                        "quality":

                        len(content)

                    }

                )

            )



            METRICS[
                "documents_collected"
            ]+=1




    logger.info(

        f"""

        SCRAPER FINISHED

        Documents={len(documents)}

        Metrics={METRICS}

        """

    )


    return documents
