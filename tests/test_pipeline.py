"""
tests/test_pipeline.py
Unit tests for the core pipeline components.
Run with:  pytest tests/ -v
"""
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.scraper.query_analyzer import analyze_query
from backend.preprocessing.cleaner import clean_text, normalize_arabic, is_arabic_heavy
from backend.preprocessing.chunker import Chunk, _split_by_paragraphs
from backend.retrieval.retriever import BM25Retriever


# ── Query Analyzer Tests ───────────────────────────────────────────────────────
class TestQueryAnalyzer:
    def test_extracts_grade(self):
        meta = analyze_query("ما هو منهج الصف الخامس الابتدائي؟")
        assert meta.grade == "الخامس الابتدائي"

    def test_extracts_subject(self):
        meta = analyze_query("علوم الصف الثالث الإعدادي")
        assert meta.subject == "علوم"

    def test_extracts_term(self):
        meta = analyze_query("منهج الترم الأول للرياضيات")
        assert meta.term == "الأول"

    def test_live_search_trigger(self):
        meta = analyze_query("ما أخبار وزارة التربية الجديدة؟")
        assert meta.needs_live_search is True

    def test_no_live_search_for_static(self):
        meta = analyze_query("ما هو منهج الصف السادس الابتدائي؟")
        assert meta.needs_live_search is False

    def test_search_query_built(self):
        meta = analyze_query("علوم الصف الخامس الترم الأول")
        assert len(meta.search_query) > 0


# ── Cleaner Tests ─────────────────────────────────────────────────────────────
class TestCleaner:
    def test_removes_html(self):
        result = clean_text("<p>مرحبًا</p>", normalize=False)
        assert "<p>" not in result
        assert "مرحبا" in result or "مرحبًا" in result

    def test_normalizes_alef(self):
        result = normalize_arabic("أإآا")
        assert result == "ااا"   # all → ا (last one already ا)

    def test_removes_harakat(self):
        result = normalize_arabic("الكِتَابُ")
        assert result == "الكتاب"

    def test_arabic_heavy(self):
        assert is_arabic_heavy("هذا نص عربي طويل نسبيًا لاختبار الدالة") is True
        assert is_arabic_heavy("This is English text only") is False


# ── Chunker Tests ─────────────────────────────────────────────────────────────
class TestChunker:
    def test_split_respects_max_chars(self):
        text = "\n\n".join(["كلمة " * 50] * 10)
        chunks = _split_by_paragraphs(text, max_chars=300)
        for chunk in chunks:
            assert len(chunk) <= 300 + 50   # slight tolerance for word boundaries

    def test_no_empty_chunks(self):
        text = "فقرة أولى\n\n\n\nفقرة ثانية\n\n"
        chunks = _split_by_paragraphs(text, max_chars=500)
        assert all(len(c.strip()) > 0 for c in chunks)


# ── BM25 Retriever Tests ──────────────────────────────────────────────────────
class TestBM25Retriever:
    def _make_chunks(self, texts):
        return [
            Chunk(
                text=t,
                chunk_id=f"test-{i:04d}",
                doc_id="test-doc",
                source_url="https://example.com",
                title="Test",
            )
            for i, t in enumerate(texts)
        ]

    def test_returns_relevant_chunk(self):
        chunks = self._make_chunks([
            "المخاليط والمحاليل في الصف الخامس",
            "الجغرافيا والتاريخ للصف الثالث",
            "العمليات الحسابية في الرياضيات",
        ])
        retriever = BM25Retriever()
        retriever.index(chunks)
        results = retriever.search("المخاليط الصف الخامس", top_k=1)
        assert len(results) >= 1
        assert "المخاليط" in results[0].chunk.text

    def test_returns_empty_on_no_match(self):
        chunks = self._make_chunks(["النص الأول", "النص الثاني"])
        retriever = BM25Retriever()
        retriever.index(chunks)
        results = retriever.search("zzz xyz", top_k=3)
        # BM25 returns 0-score results; all scores should be 0
        assert all(r.score == 0 for r in results)
