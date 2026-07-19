"""
backend/scraper/live_scraper.py
 
Live web scraper for Egyptian Ministry of Education sources.
Returns raw Document objects ready for the preprocessing pipeline.
"""
from __future__ import annotations
import re          # ✅ تم نقله للأعلى - كان داخل الدالة وبيسبب الخطأ
import time
import hashlib
from dataclasses import dataclass, field
from typing import Generator
from urllib.parse import urljoin, urlparse
 
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
    doc_type: str = "html"   # "html" | "pdf"
    metadata: dict = field(default_factory=dict)
 
    @property
    def doc_id(self) -> str:
        return hashlib.md5(self.source_url.encode()).hexdigest()[:12]
 
 
# ── HTTP client (shared, with polite headers) ──────────────────────────────────
def _make_client() -> httpx.Client:
    return httpx.Client(
        headers={
            # ✅ Headers محسّنة تحاكي المتصفح الحقيقي لتجنب الحجب
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "ar,en-US;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://www.google.com/",
            "Connection": "keep-alive",
        },
        timeout=scraper_cfg.timeout,
        follow_redirects=True,
    )
 
 
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def _fetch_url(client: httpx.Client, url: str) -> httpx.Response:
    """Fetch a URL with retry logic."""
    resp = client.get(url)
    resp.raise_for_status()
    return resp
 
 
# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf_text(pdf_bytes: bytes, source_url: str) -> RawDocument:
    """Extract text from a PDF using PyMuPDF."""
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
    """Extract meaningful text from an HTML page."""
    soup = BeautifulSoup(html, "lxml")
 
    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
 
    title = soup.title.string.strip() if soup.title else ""
 
    # ✅ الآن re متعرف عليه لأنه في أعلى الملف
    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(id=re.compile(r"content|main|body", re.I))
        or soup.body
    )
    text = main.get_text(separator="\n", strip=True) if main else soup.get_text()
 
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
 
    return RawDocument(
        content=text,
        source_url=source_url,
        title=title,
        doc_type="html",
    )
 
 
# ── DuckDuckGo fallback search ─────────────────────────────────────────────────
def _duckduckgo_search(query: str, max_results: int = 5) -> list[str]:
    """
    ✅ Fallback: DuckDuckGo search - يشتغل من Streamlit Cloud بدون API key
    بيرجع list of URLs من نتائج البحث
    """
    try:
        from duckduckgo_search import DDGS
        urls = []
        with DDGS() as ddgs:
            results = ddgs.text(
                f"site:moe.gov.eg {query}",
                max_results=max_results,
                region="eg-ar",
            )
            for r in results:
                if r.get("href"):
                    urls.append(r["href"])
        logger.info(f"DuckDuckGo found {len(urls)} URLs for: {query}")
        return urls
    except ImportError:
        logger.warning("duckduckgo_search غير مثبت — أضفه لـ requirements.txt")
        return []
    except Exception as exc:
        logger.warning(f"DuckDuckGo search failed: {exc}")
        return []
 
 
# ── Source selector ────────────────────────────────────────────────────────────
def select_sources(meta: QueryMetadata) -> list[str]:
    """Choose which MOE source URLs to scrape based on query metadata."""
    domain = meta.domain
    sources = scraper_cfg.sources.get(domain, []) + scraper_cfg.sources.get("curriculum", [])
    return list(dict.fromkeys(sources))   # deduplicate, preserve order
 
 
# ── Google-style search simulation (public search on MOE) ─────────────────────
def build_search_urls(meta: QueryMetadata) -> list[str]:
    """
    Build targeted search URLs.
    Uses MOE site-search where available; falls back to Google with site: operator.
    """
    q = meta.search_query
    return [
        f"https://www.moe.gov.eg/ar/search?q={q.replace(' ', '+')}",
        f"https://studentbooks.moe.gov.eg/search?q={q.replace(' ', '+')}",
        # public Google CSE fallback (works without API key)
        f"https://www.google.com/search?q=site:moe.gov.eg+{q.replace(' ', '+')}&num=5",
    ]
 
 
# ── Main scraper entry point ───────────────────────────────────────────────────
def scrape_for_query(meta: QueryMetadata) -> list[RawDocument]:
    """
    Scrape live documents relevant to the given query.
    Returns a list of RawDocument objects.
    """
    documents: list[RawDocument] = []
    visited: set[str] = set()
 
    urls_to_try = build_search_urls(meta) + select_sources(meta)
 
    with _make_client() as client:
        for url in urls_to_try[: scraper_cfg.max_pages * 2]:
            if url in visited:
                continue
            visited.add(url)
 
            try:
                logger.info(f"Scraping: {url}")
                resp = _fetch_url(client, url)
                content_type = resp.headers.get("content-type", "")
 
                if "pdf" in content_type or url.lower().endswith(".pdf"):
                    doc = extract_pdf_text(resp.content, url)
                else:
                    doc = extract_html_text(resp.text, url)
 
                # Filter: only keep docs with Arabic content related to query
                if len(doc.content.strip()) > 200:
                    doc.metadata.update({
                        "grade": meta.grade,
                        "subject": meta.subject,
                        "term": meta.term,
                        "domain": meta.domain,
                        "source": urlparse(url).netloc,
                    })
                    documents.append(doc)
 
                    # Follow PDF links found in HTML
                    if doc.doc_type == "html" and len(documents) < scraper_cfg.max_pages:
                        for pdf_url in _find_pdf_links(resp.text, url):
                            if pdf_url not in visited:
                                urls_to_try.append(pdf_url)
 
                time.sleep(scraper_cfg.delay)
 
            except Exception as exc:
                logger.warning(f"Failed to scrape {url}: {exc}")
 
    # ✅ Fallback: لو مفيش نتائج من المصادر المباشرة، جرّب DuckDuckGo
    if not documents:
        logger.info("No documents from direct scraping — trying DuckDuckGo fallback...")
        fallback_urls = _duckduckgo_search(meta.search_query)
 
        with _make_client() as client:
            for url in fallback_urls:
                if url in visited:
                    continue
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
                            "via": "duckduckgo_fallback",   # علامة إن الداتا جت من الـ fallback
                        })
                        documents.append(doc)
 
                    time.sleep(scraper_cfg.delay)
 
                except Exception as exc:
                    logger.warning(f"Fallback scrape failed for {url}: {exc}")
 
    logger.info(f"Scraped {len(documents)} documents")
    return documents
 
 
def _find_pdf_links(html: str, base_url: str) -> Generator[str, None, None]:
    """Find PDF links in an HTML page."""
    soup = BeautifulSoup(html, "lxml")
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf"):
            yield urljoin(base_url, href)
