"""
backend/scraper/live_scraper.py

Live web scraper — مصادر عربية مفتوحة تشتغل من Streamlit Cloud.
Primary:  Wikipedia Arabic + Archive.org
Fallback: DuckDuckGo Search (بدون API key)
"""
from __future__ import annotations
import re
import time
import hashlib
from dataclasses import dataclass, field
from typing import Generator
from urllib.parse import urljoin, urlparse, quote

import httpx
import fitz  # PyMuPDF
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import scraper_cfg
from backend.scraper.query_analyzer import QueryMetadata


@dataclass
class RawDocument:
    """A single scraped document before preprocessing."""
    content: str
    source_url: str
    title: str = ""
    doc_type: str = "html"
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return hashlib.md5(self.source_url.encode()).hexdigest()[:12]


# ── HTTP client ────────────────────────────────────────────────────────────────
def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={
            "User-Agent": scraper_cfg.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
        },
        timeout=scraper_cfg.timeout,
        follow_redirects=True,
    )


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=4))
def _fetch_url(client: httpx.Client, url: str) -> httpx.Response:
    resp = client.get(url)
    resp.raise_for_status()
    return resp


# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_bytes: bytes, source_url: str) -> RawDocument:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages_text.append(f"[صفحة {page_num + 1}]\n{text}")
    full_text = "\n\n".join(pages_text)
    title = doc.metadata.get("title", "") or urlparse(source_url).path.split("/")[-1]
    doc.close()
    return RawDocument(
        content=full_text,
        source_url=source_url,
        title=title,
        doc_type="pdf",
        metadata={"pages": len(pages_text)},
    )


# ── HTML extraction ────────────────────────────────────────────────────────────
def extract_html_text(html: str, source_url: str) -> RawDocument:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside",
                     "table.navbox", ".mw-editsection", ".reference"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else ""

    main = (
        soup.find("div", {"id": "mw-content-text"})   # ✅ Wikipedia
        or soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|body", re.I))
        or soup.body
    )
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text()
    text = re.sub(r"\n{3,}", "\n\n", text)

    return RawDocument(
        content=text,
        source_url=source_url,
        title=title,
        doc_type="html",
    )


# ── Wikipedia Arabic API ───────────────────────────────────────────────────────
def _search_wikipedia_arabic(query: str, max_results: int = 5) -> list[RawDocument]:
    """
    ✅ Wikipedia Arabic API — مفتوحة دايماً من أي سيرفر
    بتجيب نص كامل المقالات مباشرة بدون scraping
    """
    docs = []
    try:
        # خطوة 1: ابحث عن المقالات
        search_url = (
            "https://ar.wikipedia.org/w/api.php"
            f"?action=search&list=search&srsearch={quote(query)}"
            f"&srlimit={max_results}&format=json&utf8=1"
        )
        with httpx.Client(timeout=15) as client:
            resp = client.get(search_url)
            resp.raise_for_status()
            results = resp.json().get("query", {}).get("search", [])

        if not results:
            logger.info(f"Wikipedia: لا نتائج لـ '{query}'")
            return []

        # خطوة 2: جيب محتوى كل مقالة
        for item in results[:max_results]:
            page_title = item["title"]
            content_url = (
                "https://ar.wikipedia.org/w/api.php"
                f"?action=query&titles={quote(page_title)}"
                "&prop=extracts&explaintext=true&exsectionformat=plain"
                "&format=json&utf8=1"
            )
            try:
                with httpx.Client(timeout=15) as client:
                    resp = client.get(content_url)
                    resp.raise_for_status()
                    pages = resp.json().get("query", {}).get("pages", {})
                    for page_data in pages.values():
                        extract = page_data.get("extract", "")
                        if extract and len(extract) > 200:
                            page_url = f"https://ar.wikipedia.org/wiki/{quote(page_title)}"
                            docs.append(RawDocument(
                                content=extract,
                                source_url=page_url,
                                title=page_title,
                                doc_type="html",
                                metadata={"source": "ar.wikipedia.org", "via": "wikipedia_api"},
                            ))
                            logger.info(f"Wikipedia ✅ '{page_title}' — {len(extract)} chars")
            except Exception as exc:
                logger.warning(f"Wikipedia content fetch failed for '{page_title}': {exc}")

    except Exception as exc:
        logger.warning(f"Wikipedia search failed: {exc}")

    return docs


# ── DuckDuckGo fallback ────────────────────────────────────────────────────────
def _search_duckduckgo(query: str, max_results: int = 6) -> list[str]:
    """
    ✅ DuckDuckGo — بديل Google بدون API key، يشتغل من Streamlit Cloud
    بيرجع URLs فقط، الـ scraping بيحصل بعدين
    """
    try:
        from duckduckgo_search import DDGS
        urls = []
        with DDGS() as ddgs:
            # بحث عربي أولاً
            for region in scraper_cfg.ddg_regions:
                results = list(ddgs.text(
                    f"{query} التعليم المصري",
                    region=region,
                    max_results=max_results,
                ))
                for r in results:
                    if r.get("href") and r["href"] not in urls:
                        # استبعد المصادر اللي بتبلوك
                        blocked = ["moe.gov.eg", "google.com", "facebook.com", "twitter.com"]
                        if not any(b in r["href"] for b in blocked):
                            urls.append(r["href"])
                if urls:
                    break   # لقينا نتائج، مش محتاجين نكمل

        logger.info(f"DuckDuckGo ✅ وجد {len(urls)} URLs لـ '{query}'")
        return urls[:max_results]

    except ImportError:
        logger.warning("duckduckgo-search غير مثبت — أضفه لـ requirements.txt")
        return []
    except Exception as exc:
        logger.warning(f"DuckDuckGo failed: {exc}")
        return []


# ── Source selector ────────────────────────────────────────────────────────────
def _get_subject_sources(meta: QueryMetadata) -> list[str]:
    """اختار المصادر المناسبة بناءً على المادة والدومين."""
    sources = []

    # مصادر حسب المادة
    subject_map = {
        "رياضيات": "math",
        "علوم": "science",
        "فيزياء": "science",
        "كيمياء": "science",
        "أحياء": "science",
        "عربي": "arabic",
        "لغة عربية": "arabic",
        "تاريخ": "history",
        "جغرافيا": "history",
    }

    if meta.subject:
        for key, category in subject_map.items():
            if key in (meta.subject or ""):
                sources += scraper_cfg.sources.get(category, [])

    # مصادر حسب الدومين
    domain_sources = scraper_cfg.sources.get(meta.domain, [])
    sources += domain_sources

    # دايماً أضف curriculum كـ fallback
    sources += scraper_cfg.sources.get("curriculum", [])

    return list(dict.fromkeys(sources))  # deduplicate


# ── Main scraper entry point ───────────────────────────────────────────────────
def scrape_for_query(meta: QueryMetadata) -> list[RawDocument]:
    """
    Pipeline:
    1. Wikipedia Arabic API (موثوقة + عربية + مفتوحة) ✅
    2. Static sources (Wikipedia pages مخصصة للمادة) ✅
    3. DuckDuckGo fallback (لو المصادر الثابتة مش كافية) ✅
    """
    documents: list[RawDocument] = []
    visited: set[str] = set()

    # ══════════════════════════════════════════════════════
    # الخطوة 1: Wikipedia Arabic API (الأسرع والأأمن)
    # ══════════════════════════════════════════════════════
    logger.info(f"🔍 Wikipedia API: '{meta.search_query}'")
    wiki_docs = _search_wikipedia_arabic(meta.search_query, max_results=4)
    for doc in wiki_docs:
        doc.metadata.update({
            "grade": meta.grade,
            "subject": meta.subject,
            "term": meta.term,
            "domain": meta.domain,
        })
        documents.append(doc)
        visited.add(doc.source_url)

    # ══════════════════════════════════════════════════════
    # الخطوة 2: Static sources (Wikipedia pages مخصصة)
    # ══════════════════════════════════════════════════════
    static_urls = _get_subject_sources(meta)

    with _make_client() as client:
        for url in static_urls[: scraper_cfg.max_pages]:
            if url in visited:
                continue
            visited.add(url)

            try:
                logger.info(f"Scraping static: {url}")
                resp = _fetch_url(client, url)
                content_type = resp.headers.get("content-type", "")

                if "pdf" in content_type or url.lower().endswith(".pdf"):
                    doc = extract_pdf_text(resp.content, url)
                else:
                    doc = extract_html_text(resp.text, url)

                if len(doc.content.strip()) > 200:
                    doc.metadata.update({
                        "grade": meta.grade,
                        "subject": meta.subject,
                        "term": meta.term,
                        "domain": meta.domain,
                        "source": urlparse(url).netloc,
                    })
                    documents.append(doc)

                    # تتبع روابط PDF
                    if doc.doc_type == "html" and len(documents) < scraper_cfg.max_pages:
                        for pdf_url in _find_pdf_links(resp.text, url):
                            if pdf_url not in visited:
                                static_urls.append(pdf_url)

                time.sleep(scraper_cfg.delay)

            except Exception as exc:
                logger.warning(f"Failed static scrape {url}: {exc}")

    # ══════════════════════════════════════════════════════
    # الخطوة 3: DuckDuckGo fallback (لو محتاجين أكتر)
    # ══════════════════════════════════════════════════════
    if len(documents) < 2:
        logger.info("📡 DuckDuckGo fallback — المصادر الثابتة غير كافية...")
        ddg_urls = _search_duckduckgo(meta.search_query)

        with _make_client() as client:
            for url in ddg_urls:
                if url in visited or len(documents) >= scraper_cfg.max_pages:
                    break
                visited.add(url)

                try:
                    logger.info(f"Fallback scraping: {url}")
                    resp = _fetch_url(client, url)
                    content_type = resp.headers.get("content-type", "")

                    if "pdf" in content_type or url.lower().endswith(".pdf"):
                        doc = extract_pdf_text(resp.content, url)
                    else:
                        doc = extract_html_text(resp.text, url)

                    if len(doc.content.strip()) > 200:
                        doc.metadata.update({
                            "grade": meta.grade,
                            "subject": meta.subject,
                            "term": meta.term,
                            "domain": meta.domain,
                            "source": urlparse(url).netloc,
                            "via": "duckduckgo",
                        })
                        documents.append(doc)

                    time.sleep(scraper_cfg.delay)

                except Exception as exc:
                    logger.warning(f"Fallback scrape failed {url}: {exc}")

    logger.info(f"✅ Scraped {len(documents)} documents total")
    return documents


def _find_pdf_links(html: str, base_url: str) -> Generator[str, None, None]:
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            yield urljoin(base_url, href)
