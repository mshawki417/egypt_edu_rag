"""
Production Live Scraper
Real-Time Education RAG Source Collector

Version:
Advanced v5

Features:
- Parallel multi-source scraping
- Curriculum-aware retrieval
- Source credibility scoring
- Citation-ready metadata
- Async optimized pipeline
- Duplicate removal
- RAG optimized documents

Compatible with:
- query_analyzer.py v3+
- orchestrator.py
- chunker.py
- retriever.py
"""


from __future__ import annotations


import asyncio

import hashlib

import os

import re

import time


from dataclasses import dataclass, field


from urllib.parse import (
    urlparse,
    urlunparse
)


from collections import defaultdict


import httpx


import fitz


from bs4 import BeautifulSoup


from cachetools import TTLCache


from loguru import logger


from ddgs import DDGS




# =====================================================
# CONFIGURATION
# =====================================================


class ScraperConfig:
    """
    Central scraper configuration.
    Supports ENV overrides.
    """


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
            3000
        )
    )


    MAX_RETRIES = int(
        os.getenv(
            "SCRAPER_RETRIES",
            3
        )
    )


    MAX_RESULTS = int(
        os.getenv(
            "SCRAPER_RESULTS",
            20
        )
    )

    MAX_CONCURRENT_REQUESTS = int(
        os.getenv(
            "SCRAPER_CONCURRENT",
            8
        )
    )


    MAX_DOCUMENT_SIZE = int(
        os.getenv(
            "MAX_DOCUMENT_SIZE",
            600000
        )
    )


    MIN_CONTENT_LENGTH = int(
        os.getenv(
            "MIN_CONTENT_LENGTH",
            250
        )
    )


    USER_AGENT = (
    
        "Mozilla/5.0 "
    
        "(Windows NT 10.0; Win64; x64) "
    
        "AppleWebKit/537.36 "
    
        "(KHTML, like Gecko) "
    
        "Chrome/120.0 Safari/537.36 "
    
        "EgyptEducationRAG/5.0"
    
    )


# =====================================================
# SOURCE CREDIBILITY MODEL
# =====================================================


SOURCE_CREDIBILITY = {


    # Official Ministry Sources

    "moe.gov.eg": 1.00,


    "ekb.eg": 0.95,


    "study.ekb.eg": 0.95,



    # Educational Knowledge

    "marefa.org": 0.80,


    "wikipedia.org": 0.65,



    # News Sources

    "youm7.com": 0.45,


    "elwatannews.com": 0.45

}





BLOCKED_DOMAINS = [

    "facebook.com",

    "instagram.com",

    "tiktok.com",

    "reddit.com",

    "pinterest.com"

]





ALLOWED_DOMAINS = list(

    SOURCE_CREDIBILITY.keys()

)


# =====================================================
# METRICS
# =====================================================


METRICS = {


    "documents_collected": 0,


    "fetch_success": 0,


    "fetch_failed": 0,


    "fetch_errors": 0,


    "cache_hits": 0,


    "robots_blocked": 0,


    "blocked_urls": 0,


    "search_requests": 0,


    "search_results": 0,


    "duplicate_removed": 0

}


# =====================================================
# DATA MODELS
# =====================================================


@dataclass
class ScrapedContent:
    """
    Internal extraction object.
    """


    text: str


    title: str = ""


    url: str = ""


    doc_type: str = "web"


    source_score: float = 0.0


    content_hash: str = ""



    metadata: dict = field(

        default_factory=dict

    )



@dataclass
class RawDocument:
    """
    Final document sent to RAG pipeline.
    """


    content: str


    source_url: str


    title: str = ""


    doc_id: str = ""


    doc_type: str = "web"



    metadata: dict = field(

        default_factory=dict

    )



# =====================================================
# CACHE SYSTEM
# =====================================================


SCRAPER_CACHE = TTLCache(

    maxsize=ScraperConfig.CACHE_SIZE,

    ttl=ScraperConfig.CACHE_TTL

)


def generate_cache_key(
    url: str
) -> str:
    """
    Generate stable cache key.
    """


    return hashlib.sha256(

        url.encode(
            "utf-8"
        )

    ).hexdigest()


def generate_content_hash(
    text: str
) -> str:
    """
    Generate document fingerprint.
    Used for duplicate removal.
    """


    return hashlib.sha256(

        text[:15000]
        .encode(
            "utf-8"
        )

    ).hexdigest()


# =====================================================
# URL HELPERS
# =====================================================


def sanitize_log(
    value
) -> str:
    """
    Prevent unsafe logs.
    """


    if not value:

        return ""


    return re.sub(

        r"[\r\n\t]",

        " ",

        str(value)

    )[:300]


def normalize_url(
    url: str
) -> str:
    """
    Normalize URL format.
    """


    if not url:

        return ""



    url = url.strip()



    if not url.startswith(
        (
            "http://",
            "https://"
        )
    ):

        url = "https://" + url


    try:

        parsed = urlparse(
            url
        )


        return urlunparse(

            (

                parsed.scheme.lower(),

                parsed.netloc.lower(),

                parsed.path or "/",

                "",

                parsed.query,

                ""

            )

        )



    except Exception:


        return ""


# =====================================================
# ASYNC LIMITER
# =====================================================


_HOST_LIMITERS = defaultdict(

    lambda: asyncio.Semaphore(

        ScraperConfig.MAX_CONCURRENT_REQUESTS

    )

)


def get_host_limiter(
    host: str
):

    return _HOST_LIMITERS[host]

# =====================================================
# DOMAIN VALIDATION
# =====================================================


def is_allowed_domain(
    url: str,
    allowed_domains=None
) -> bool:
    """
    Validate URL source.

    Protection against:
    - fake domains
    - malicious sources
    - unsupported websites
    """


    try:

        normalized = normalize_url(
            url
        )


        parsed = urlparse(
            normalized
        )


        domain = parsed.netloc.lower()



        if domain.startswith(
            "www."
        ):

            domain = domain[4:]



        if not domain:

            return False



        # Block unwanted platforms

        for blocked in BLOCKED_DOMAINS:

            if domain.endswith(
                blocked
            ):

                METRICS[
                    "blocked_urls"
                ] += 1


                return False




        domains = (
            allowed_domains
            or
            ALLOWED_DOMAINS
        )



        return any(

            domain.endswith(
                allowed
            )

            for allowed in domains

        )



    except Exception as e:


        logger.error(

            f"Domain validation error: {sanitize_log(e)}"

        )


        return False



# Backward compatibility

def valid_url(
    url:str
):

    return is_allowed_domain(
        url
    )


# =====================================================
# ROBOTS HANDLING
# =====================================================


async def allowed_by_robots(
    client,
    url:str
)->bool:
    """
    Basic robots.txt compliance.

    Prevents scraping restricted paths.
    """

    try:


        parsed = urlparse(
            url
        )


        robots_url = (

            f"{parsed.scheme}"

            f"://{parsed.netloc}"

            "/robots.txt"

        )


        response = await client.get(

            robots_url,

            timeout=5

        )


        if response.status_code != 200:

            return True


        path = parsed.path.lower()



        user_agent_section = False



        for line in response.text.splitlines():

            line = line.strip().lower()



            if line.startswith(
                "user-agent"
            ):


                user_agent_section = (

                    "*"
                    in line

                )


            if (

                user_agent_section

                and

                line.startswith(
                    "disallow:"
                )

            ):


                blocked_path = (

                    line.split(
                        ":",
                        1
                    )[1]

                    .strip()

                )



                if blocked_path and path.startswith(
                    blocked_path
                ):


                    METRICS[
                        "robots_blocked"
                    ] += 1



                    return False



        return True




    except Exception:


        # Fail open

        return True



# =====================================================
# CONTENT CLEANING
# =====================================================


def clean_text(
    text:str
)->str:
    """
    Normalize extracted text.

    Used for:
    - HTML
    - PDF
    - RAG chunking
    """



    if not text:

        return ""



    text = re.sub(

        r"\s+",

        " ",

        text

    )



    text = re.sub(

        r"\n{3,}",

        "\n\n",

        text

    )



    text = text.strip()



    return text


# =====================================================
# HTML EXTRACTION
# =====================================================


def extract_text_from_html(
    html:str,
    url:str=""
)->ScrapedContent:
    """
    Extract educational content from HTML.

    Keeps:
    - title
    - description
    - paragraphs

    Removes:
    - scripts
    - menus
    - ads
    """



    try:


        soup = BeautifulSoup(

            html,

            "html.parser"

        )



        title = ""



        if soup.title:

            title = soup.title.text.strip()





        description = ""



        meta = soup.find(

            "meta",

            attrs={

                "name":

                "description"

            }

        )



        if meta:

            description = meta.get(

                "content",

                ""

            )






        remove_tags = [

            "script",

            "style",

            "nav",

            "footer",

            "header",

            "aside",

            "form",

            "noscript",

            "svg"

        ]



        for tag in soup(
            remove_tags
        ):

            tag.decompose()





        paragraphs = []



        for element in soup.find_all(

            [

                "p",

                "article",

                "section"

            ]

        ):


            text = element.get_text(

                " ",

                strip=True

            )



            if len(text) > 50:


                paragraphs.append(

                    text

                )






        body = "\n".join(

            paragraphs

        )




        final_content = clean_text(

            (

                title

                +

                "\n"

                +

                description

                +

                "\n"

                +

                body

            )

        )





        return ScrapedContent(

            text=final_content,

            title=title,

            url=url,

            doc_type="web",

            source_score=calculate_domain_score(

                url

            ),

            content_hash=generate_content_hash(

                final_content

            )

        )




    except Exception as e:


        logger.error(

            f"HTML extraction failed: {sanitize_log(e)}"

        )


        return ScrapedContent(

            text=""

        )



# =====================================================
# PDF EXTRACTION
# =====================================================


def extract_pdf_text(
    data:bytes,
    url:str=""
)->ScrapedContent:
    """
    Extract text from PDF files.

    Optimized for:
    - Books
    - Curriculum PDFs
    - Educational documents
    """



    try:


        pdf = fitz.open(

            stream=data,

            filetype="pdf"

        )



        pages = []



        for page_number, page in enumerate(pdf):


            text = page.get_text(

                "text"

            )



            if text:


                pages.append(

                    text

                )



        pdf.close()



        content = clean_text(

            "\n".join(

                pages

            )

        )


        return ScrapedContent(

            text=content,

            url=url,

            doc_type="pdf",

            source_score=calculate_domain_score(

                url

            ),

            content_hash=generate_content_hash(

                content

            )

        )




    except Exception as e:


        logger.error(

            f"PDF extraction failed: {sanitize_log(e)}"

        )


        return ScrapedContent(

            text=""

        )


# =====================================================
# CONTENT VALIDATION
# =====================================================


def validate_content(
    content:str
)->bool:
    """
    Validate extracted document.

    Reject:
    - empty pages
    - tiny texts
    - huge dumps
    """



    if not content:

        return False



    length = len(

        content

    )



    if length < ScraperConfig.MIN_CONTENT_LENGTH:

        return False



    if length > ScraperConfig.MAX_DOCUMENT_SIZE:

        return False




    # Remove pages without Arabic/English content

    useful_chars = len(

        re.findall(

            r"[A-Za-z\u0600-\u06FF]",

            content

        )

    )



    if useful_chars < 100:

        return False



    return True

# =====================================================
# SEARCH QUERY EXPANSION
# =====================================================


def build_search_queries(meta):
    """
    Generate multiple educational search strategies.

    Example:

    Input:
        شرح الجهاز الهضمي الصف الثاني الاعدادي

    Output:
        [
            original query,
            topic explanation,
            ministry book,
            EKB query,
            grade query
        ]
    """

    queries = []


    base_query = getattr(
        meta,
        "search_query",
        ""
    )


    if base_query:
        queries.append(
            base_query
        )



    topic = getattr(
        meta,
        "topic",
        None
    )


    subject = getattr(
        meta,
        "subject",
        None
    )


    grade = getattr(
        meta,
        "grade",
        None
    )



    if topic:


        queries.extend(

            [

                f"{topic} شرح",

                f"{topic} منهج مصر",

                f"{topic} كتاب الوزارة",

                f"{topic} بنك المعرفة المصري",

            ]

        )



    if subject and topic:


        queries.append(

            f"{subject} {topic} شرح"

        )



    if grade and subject:


        queries.append(

            (

                f"{subject} "

                f"{grade} "

                "وزارة التربية والتعليم"

            )

        )



    keywords = getattr(

        meta,

        "keywords",

        []

    )


    for keyword in keywords:


        queries.append(

            f"{keyword} شرح تعليمي"

        )



    return list(

        dict.fromkeys(

            [

                q.strip()

                for q in queries

                if q and q.strip()

            ]

        )

    )



# =====================================================
# DUCKDUCKGO SEARCH LAYER
# =====================================================


def search_engine(
    query:str
):

    """
    DuckDuckGo search wrapper.

    Returns only trusted educational URLs.
    """


    urls=[]


    try:


        search_query = (

            f"{query} "

            "مصر "

            "وزارة التربية والتعليم"

        )



        with DDGS() as ddgs:


            results = ddgs.text(

                search_query,

                region="eg-ar",

                safesearch="moderate",

                max_results=ScraperConfig.MAX_RESULTS

            )



            for item in results:


                url=item.get(
                    "href"
                )


                if not url:

                    continue



                normalized = normalize_url(

                    url

                )


                if is_allowed_domain(

                    normalized

                ):


                    urls.append(

                        normalized

                    )


        METRICS["search_requests"] += 1

        METRICS["search_results"] += len(urls)


    except Exception as e:


        logger.error(

            f"Search engine error: "

            f"{sanitize_log(e)}"

        )



    return list(

        dict.fromkeys(

            urls

        )

    )

def build_search_urls(query: str):
    """
    Backward compatible search wrapper.

    Used by async pipeline.
    """

    METRICS["search_requests"] += 1

    return search_engine(query)


# =====================================================
# PARALLEL SEARCH EXECUTION
# =====================================================


async def parallel_search(
    meta
):

    """
    Execute multiple educational
    searches concurrently.
    """


    queries = build_search_queries(

        meta

    )


    tasks = [

        asyncio.to_thread(

            search_engine,

            query

        )

        for query in queries

    ]



    results = await asyncio.gather(

        *tasks,

        return_exceptions=True

    )



    urls=[]



    for result in results:


        if isinstance(

            result,

            list

        ):


            urls.extend(

                result

            )



    return list(

        dict.fromkeys(

            urls

        )

    )



# =====================================================
# SOURCE RANKING
# =====================================================

def calculate_source_score(
    url:str
):

    """
    Rank source credibility.

    Higher score =
    Better RAG source.
    """


    domain = (

        urlparse(url)

        .netloc

        .lower()

        .replace(
            "www.",
            ""
        )

    )


    score = 0.30



    for source,value in SOURCE_CREDIBILITY.items():


        if domain.endswith(

            source

        ):


            score=value

            break



    return round(

        score,

        2

    )



# ADD THIS HERE

def calculate_domain_score(
    url: str
):
    """
    Compatibility wrapper.
    Used by extraction functions.
    """

    return calculate_source_score(
        url
    )



def source_priority_score(
    content:ScrapedContent
):

    """
    Final ranking score.

    Combines:

    - Authority
    - Length
    - Content quality
    """

# =====================================================
# PARAGRAPH SPLITTING
# =====================================================


def split_paragraphs(
    text:str
):


    """
    Split content into
    semantic chunks.
    """


    paragraphs = re.split(

        r"\n+|(?<=[.!؟])\s+",

        text

    )


    return [

        p.strip()

        for p in paragraphs

        if len(
            p.strip()
        ) >= 50

    ]







# =====================================================
# EDUCATIONAL RELEVANCE SCORING
# =====================================================


def score_paragraph(
    paragraph:str,
    meta
):


    """
    Educational relevance model.


    Score factors:

    Topic      40%

    Subject    20%

    Grade      20%

    Keywords   20%

    """


    score = 0.0



    text = paragraph.lower()



    topic = getattr(

        meta,

        "topic",

        None

    )


    subject = getattr(

        meta,

        "subject",

        None

    )


    grade = getattr(

        meta,

        "grade",

        None

    )




    if topic:

        if topic.lower() in text:

            score += 0.40





    if subject:


        if subject.lower() in text:

            score += 0.20





    if grade:


        if grade.lower() in text:

            score += 0.20






    keywords = getattr(

        meta,

        "keywords",

        []

    )



    if keywords:


        matched=sum(

            1

            for k in keywords

            if k.lower() in text

        )



        score += min(

            matched * 0.05,

            0.20

        )



    return min(

        score,

        1.0

    )






# =====================================================
# PARAGRAPH RANKING
# =====================================================


def rank_content(
    content:ScrapedContent,
    meta
):


    """
    Keep only most relevant
    educational paragraphs.
    """


    paragraphs = split_paragraphs(

        content.text

    )



    ranked=[]



    for paragraph in paragraphs:


        relevance = score_paragraph(

            paragraph,

            meta

        )


        ranked.append(

            (

                relevance,

                paragraph

            )

        )




    ranked.sort(

        key=lambda x:x[0],

        reverse=True

    )



    selected=[


        paragraph

        for score,paragraph

        in ranked[:25]

    ]



    content.text="\n\n".join(

        selected

    )



    content.metadata.update(

        {

            "paragraph_count":

            len(paragraphs),


            "selected_paragraphs":

            len(selected),


            "top_relevance":

            ranked[0][0]

            if ranked

            else 0

        }

    )



    return content

# =====================================================
# HTTP CLIENT FACTORY
# =====================================================


def create_http_client():

    """
    Production HTTP client.

    Features:
    - Connection pooling
    - Keep alive
    - Redirect support
    """

    limits = httpx.Limits(

        max_connections=30,

        max_keepalive_connections=15

    )


    timeout = httpx.Timeout(

        connect=10,

        read=ScraperConfig.REQUEST_TIMEOUT,

        write=10,

        pool=10

    )


    return httpx.AsyncClient(

        headers={

            "User-Agent":

            ScraperConfig.USER_AGENT

        },

        timeout=timeout,

        limits=limits,

        follow_redirects=True

    )





# =====================================================
# CACHE HELPERS
# =====================================================


def get_cached_content(url):


    key = generate_cache_key(

        url

    )


    if key in SCRAPER_CACHE:


        METRICS[

            "cache_hits"

        ] += 1


        return SCRAPER_CACHE[key]


    return None





def save_cache(
    url,
    content
):


    key = generate_cache_key(

        url

    )


    SCRAPER_CACHE[key]=content





# =====================================================
# ROBOTS ADVANCED CHECK
# =====================================================


ROBOTS_CACHE = TTLCache(

    maxsize=500,

    ttl=3600

)




async def allowed_by_robots(
    client,
    url
):


    """
    Basic robots.txt parser.

    Cached per domain.
    """



    parsed=urlparse(

        url

    )


    domain=parsed.netloc



    if domain in ROBOTS_CACHE:


        return ROBOTS_CACHE[domain]




    try:


        robots_url=(

            f"{parsed.scheme}"

            f"://{domain}"

            "/robots.txt"

        )



        response=await client.get(

            robots_url

        )



        if response.status_code != 200:


            ROBOTS_CACHE[domain]=True

            return True





        path=parsed.path.lower()



        allowed=True



        for line in response.text.splitlines():


            line=line.lower().strip()



            if line.startswith(

                "disallow:"

            ):


                blocked=line.split(

                    ":",

                    1

                )[1].strip()



                if blocked and path.startswith(

                    blocked

                ):


                    allowed=False

                    break



        ROBOTS_CACHE[domain]=allowed



        return allowed



    except Exception:


        return True







# =====================================================
# CONTENT DEDUPLICATION
# =====================================================


CONTENT_HASHES=set()




def is_duplicate_content(
    text
):


    """
    Prevent duplicate documents
    entering RAG index.
    """


    content_hash = generate_content_hash(

        text

    )



    if content_hash in CONTENT_HASHES:


        return True



    CONTENT_HASHES.add(

        content_hash

    )


    return False






# =====================================================
# RESPONSE CONTENT TYPE DETECTOR
# =====================================================


def detect_document_type(
    content_type,
    url
):


    content_type=content_type.lower()



    if (

        "application/pdf"

        in content_type

        or

        url.lower().endswith(

            ".pdf"

        )

    ):

        return "pdf"



    if "text/html" in content_type:


        return "web"



    return None






# =====================================================
# FETCH ENGINE
# =====================================================


async def fetch_url(
    client,
    url,
    meta
):


    """
    Production async fetch.

    Pipeline:

    URL
     |
    Validation
     |
    Cache
     |
    Robots
     |
    HTTP
     |
    Extraction
     |
    Cleaning
     |
    Ranking
     |
    Metadata
    """



    url=normalize_url(

        url

    )



    if not is_allowed_domain(

        url

    ):


        return None





    cached=get_cached_content(

        url

    )



    if cached:


        return cached






    host=urlparse(

        url

    ).netloc



    limiter=get_host_limiter(

        host

    )



    async with limiter:



        for attempt in range(

            ScraperConfig.MAX_RETRIES

        ):



            try:


                start=time.time()



                if not await allowed_by_robots(

                    client,

                    url

                ):


                    METRICS[

                        "robots_blocked"

                    ] += 1


                    return None





                response=await client.get(

                    url

                )



                response.raise_for_status()



                document_type=detect_document_type(

                    response.headers.get(

                        "content-type",

                        ""

                    ),

                    url

                )



                if not document_type:


                    return None





                if document_type=="pdf":


                    content=extract_pdf_text(

                        response.content,

                        url

                    )


                else:


                    content=extract_text_from_html(

                        response.text,

                        url

                    )





                if not validate_content(

                    content.text

                ):


                    return None





                if is_duplicate_content(

                    content.text

                ):


                    logger.info(

                        f"Duplicate skipped {url}"

                    )


                    return None






                content=rank_content(

                    content,

                    meta

                )





                content.metadata.update(

                    {

                        "url":

                        url,


                        "source":

                        urlparse(url).netloc,


                        "source_score":

                        calculate_source_score(url),


                        "document_type":

                        document_type,


                        "fetch_time":

                        round(

                            time.time()

                            -

                            start,

                            3

                        ),


                        "char_count":

                        len(content.text),


                        "content_hash":

                        content.content_hash


                    }

                )





                save_cache(

                    url,

                    content

                )





                METRICS[

                    "fetch_success"

                ] += 1




                logger.info(

                    (

                        "FETCH OK | "

                        f"{url} | "

                        f"{document_type} | "

                        f"{len(content.text)} chars"

                    )

                )



                return content





            except (

                httpx.TimeoutException,

                httpx.NetworkError

            ) as e:




                wait=(

                    2 ** attempt

                )



                logger.warning(

                    (

                        f"Retry "

                        f"{attempt+1}/"

                        f"{ScraperConfig.MAX_RETRIES}"

                        f" {url}"

                    )

                )



                await asyncio.sleep(

                    wait

                )





            except httpx.HTTPStatusError as e:



                METRICS[

                    "fetch_errors"

                ] += 1



                logger.warning(

                    (

                        f"HTTP error "

                        f"{e.response.status_code}"

                    )

                )


                return None





            except Exception as e:



                METRICS[

                    "fetch_errors"

                ] += 1



                logger.error(

                    (

                        "Fetch failed "

                        f"{sanitize_log(e)}"

                    )

                )


                return None



    return None






# =====================================================
# FINAL RAG DOCUMENT BUILDER
# =====================================================


def build_rag_document(
    content:ScrapedContent,
    query:str
):


    """
    Convert scraped content
    into RAG compatible object.
    """



    doc_id=hashlib.sha256(

        (

            content.url

            +

            content.content_hash

        )

        .encode()

    ).hexdigest()





    return RawDocument(

        content=content.text,


        source_url=content.url,


        title=content.title,


        doc_id=doc_id,


        doc_type=content.doc_type,


        metadata={

            **content.metadata,


            "query":

            query,


            "citation":

            {

                "source":

                content.url,


                "title":

                content.title

            }

        }

    )

# =====================================================
# FINAL DOCUMENT BUILDER
# =====================================================


def create_document_id(
    url:str,
    content:str
)->str:


    """
    Stable document identifier
    """

    payload = (

        url +

        content[:500]

    )


    return hashlib.sha256(

        payload.encode(
            "utf-8"
        )

    ).hexdigest()





def build_final_metadata(
    content:ScrapedContent,
    query:str
)->dict:


    """
    Citation-ready metadata
    """


    return {


        "source_url":

        content.url,


        "source":

        urlparse(
            content.url
        ).netloc,



        "document_type":

        content.doc_type,



        "source_score":

        content.source_score,



        "content_hash":

        content.content_hash,



        "query":

        query,



        "title":

        content.title,



        "content_length":

        len(
            content.text
        ),



        "created_at":

        time.time()

    }







# =====================================================
# DUPLICATE FILTER
# =====================================================


def remove_duplicate_documents(
    documents:list[RawDocument]
):


    """
    Remove duplicated documents
    using content hash.
    """


    seen=set()


    unique=[]


    for doc in documents:


        fingerprint=hashlib.sha256(

            doc.content[:1000]
            .encode(
                "utf-8"
            )

        ).hexdigest()



        if fingerprint in seen:

            continue



        seen.add(
            fingerprint
        )


        unique.append(
            doc
        )



    return unique








# =====================================================
# METRICS REPORT
# =====================================================


def scraper_report()->dict:


    """
    Production monitoring output
    """


    return {


        "documents":

        METRICS[
            "documents_collected"
        ],


        "fetch_success":

        METRICS[
            "fetch_success"
        ],


        "errors":

        METRICS[
            "fetch_errors"
        ],


        "cache_hits":

        METRICS[
            "cache_hits"
        ],


        "blocked":

        METRICS[
            "blocked_urls"
        ],


        "robots_blocked":

        METRICS[
            "robots_blocked"
        ],


        "search_results":

        METRICS[
            "search_results"
        ]

    }







# =====================================================
# FINAL ASYNC PIPELINE
# =====================================================


async def async_scrape_for_query(
    meta
)->list[RawDocument]:


    """
    Main production RAG scraper.

    Pipeline:

    Query Analyzer

          |

    Query Expansion

          |

    Parallel Search

          |

    URL Filtering

          |

    Async Fetch Pool

          |

    HTML/PDF Extraction

          |

    Ranking

          |

    Metadata Creation

          |

    Deduplication

          |

    RAG Documents


    """



    start=time.time()



    logger.info(

        f"""
START SCRAPER

Query:
{meta.search_query}

"""

    )



    # -------------------------------------------------
    # 1) Build search queries
    # -------------------------------------------------


    queries = build_search_queries(
        meta
    )



    logger.info(

        f"Generated queries: {len(queries)}"

    )




    # -------------------------------------------------
    # 2) Parallel Search
    # -------------------------------------------------


    search_tasks=[


        asyncio.to_thread(

            build_search_urls,

            query

        )

        for query in queries

    ]




    search_results = await asyncio.gather(

        *search_tasks,

        return_exceptions=True

    )




    urls=[]



    for result in search_results:


        if isinstance(

            result,

            list

        ):


            urls.extend(

                result

            )





    urls=list(

        dict.fromkeys(

            urls

        )

    )




    if not urls:


        logger.warning(

            "No URLs discovered"

        )


        return []





    logger.info(

        f"URLs collected: {len(urls)}"

    )






    # -------------------------------------------------
    # 3) Async HTTP Fetch Pool
    # -------------------------------------------------


    documents=[]



    async with httpx.AsyncClient(

        headers={

            "User-Agent":

            ScraperConfig.USER_AGENT

        },

        timeout=

        ScraperConfig.REQUEST_TIMEOUT,


        follow_redirects=True,


        http2=True,


        limits=httpx.Limits(

            max_connections=30,

            max_keepalive_connections=15

        )


    ) as client:




        fetch_jobs=[


            fetch_url(

                client,

                url,

                meta

            )


            for url in urls

        ]





        responses=await asyncio.gather(

            *fetch_jobs,

            return_exceptions=True

        )






        # -------------------------------------------------
        # 4) Build RAG Documents
        # -------------------------------------------------


        for response in responses:



            if not response:

                continue



            if isinstance(

                response,

                Exception

            ):

                continue




            if not response.text:

                continue




            doc_id=create_document_id(

                response.url,

                response.text

            )




            metadata=build_final_metadata(

                response,

                meta.search_query

            )





            documents.append(

                RawDocument(


                    content=response.text,


                    source_url=response.url,


                    title=response.title,


                    doc_id=doc_id,


                    doc_type=response.doc_type,


                    metadata=metadata

                )

            )








    # -------------------------------------------------
    # 5) Remove duplicates
    # -------------------------------------------------


    documents = remove_duplicate_documents(

        documents

    )





    METRICS[

        "documents_collected"

    ] += len(documents)







    # -------------------------------------------------
    # 6) Final Logging
    # -------------------------------------------------


    elapsed=time.time()-start



    logger.info(

        f"""

================================================

SCRAPER COMPLETED


Documents:
{len(documents)}


Time:
{elapsed:.2f}s


Metrics:
{scraper_report()}


================================================

"""

    )




    return documents







# =====================================================
# PRODUCTION ENTRY POINT
# =====================================================


async def run_scraper(
    meta
):


    """
    External entry point.

    Used by:

    orchestrator.py

    Example:


    documents = await run_scraper(meta)


    """



    try:


        return await async_scrape_for_query(

            meta

        )



    except Exception as e:


        logger.exception(

            f"Production scraper failed: {e}"

        )


        return []






# =====================================================
# TEST MODE
# =====================================================


if __name__ == "__main__":


    import asyncio



    class TestQuery:


        search_query="الجهاز الهضمي الصف الثاني الاعدادي"


        topic="الجهاز الهضمي"


        subject="علوم"



        grade="الثاني الاعدادي"


        keywords=[

            "الهضم",

            "الأمعاء",

            "المعدة"

        ]




    results=asyncio.run(

        run_scraper(

            TestQuery()

        )

    )



    print(

        f"""

Documents collected:
{len(results)}

"""

    )


    for doc in results[:3]:


        print(

            "="*50

        )


        print(

            doc.title

        )


        print(

            doc.source_url

        )


        print(

            doc.metadata

        )

