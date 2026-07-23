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
    metadata: dict = field(default_factory=dict)


# ==========================
# Cache
# ==========================

SCRAPER_CACHE = TTLCache(maxsize=1000, ttl=3600)


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
    if any(domain in url for domain in BLOCKED_DOMAINS):
        return False

    return any(domain in url for domain in ALLOWED_DOMAINS)


# ==========================
# Fetch
# ==========================

async def fetch_url(client, url):
    if url in SCRAPER_CACHE:
        return SCRAPER_CACHE[url]

    semaphore = get_request_limit()

    async with semaphore:
        try:
            response = await client.get(url, timeout=20)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "")

            if "pdf" in content_type.lower():
                pdf = fitz.open(stream=response.content, filetype="pdf")

                text = ""
                for page in pdf:
                    text += page.get_text()

                pdf.close()

            else:
                soup = BeautifulSoup(response.text, "html.parser")

                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                text = soup.get_text(separator=" ", strip=True)

            if len(text) < 200:
                return None

            SCRAPER_CACHE[url] = text
            return text

        except Exception as e:
            logger.warning(f"Fetch failed {url}: {e}")
            return None


# ==========================
# Search
# ==========================

def build_search_urls(query):
    logger.info(f"DDG Search: {query}")

    try:
        search_query = f"{query} وزارة التربية والتعليم مصر"
        urls = []

        with DDGS() as ddgs:
            results = ddgs.text(
                search_query,
                region="eg-ar",
                safesearch="moderate",
                max_results=10
            )

            for item in results:
                url = item.get("href")

                if url and valid_url(url):
                    urls.append(url)

        logger.info(f"URLs found: {len(urls)}")
        return urls

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return []


# ==========================
# Main Scraper
# ==========================

async def async_scrape_for_query(meta):
    query = meta.search_query
    logger.info(f"Scraping query: {query}")

    urls = await asyncio.to_thread(build_search_urls, query)

    if not urls:
        return []

    documents = []

    async with httpx.AsyncClient(
        headers={"User-Agent": "Mozilla/5.0"},
        follow_redirects=True
    ) as client:

        results = await asyncio.gather(
            *[fetch_url(client, url) for url in urls],
            return_exceptions=True
        )

        for url, content in zip(urls, results):
            if not content:
                continue

            if isinstance(content, Exception):
                continue

            doc_id = hashlib.md5(url.encode()).hexdigest()

            documents.append(
                RawDocument(
                    content=content,
                    source_url=url,
                    title=query,
                    doc_id=doc_id,
                    metadata={"query": query}
                )
            )

    logger.info(f"Documents collected: {len(documents)}")
    return documents
