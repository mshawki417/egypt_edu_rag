"""
backend/scraper/query_analyzer.py
Parses Arabic/mixed questions and extracts structured metadata.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from loguru import logger


GRADE_PATTERNS: dict[str, str] = {
    r"الأول\s*الابتدائي|الصف\s*الأول\s*ابتدائي|grade\s*1": "الأول الابتدائي",
    r"الثاني\s*الابتدائي|الصف\s*الثاني\s*ابتدائي|grade\s*2": "الثاني الابتدائي",
    r"الثالث\s*الابتدائي|grade\s*3": "الثالث الابتدائي",
    r"الرابع\s*الابتدائي|grade\s*4": "الرابع الابتدائي",
    r"الخامس\s*الابتدائي|grade\s*5": "الخامس الابتدائي",
    r"السادس\s*الابتدائي|grade\s*6": "السادس الابتدائي",
    r"الأول\s*الإعدادي|grade\s*7": "الأول الإعدادي",
    r"الثاني\s*الإعدادي|grade\s*8": "الثاني الإعدادي",
    r"الثالث\s*الإعدادي|grade\s*9": "الثالث الإعدادي",
    r"الأول\s*الثانوي|grade\s*10": "الأول الثانوي",
    r"الثاني\s*الثانوي|grade\s*11": "الثاني الثانوي",
    r"الثالث\s*الثانوي|grade\s*12|الثانوية\s*العامة": "الثالث الثانوي",
}

SUBJECT_PATTERNS: dict[str, str] = {
    r"علوم|science": "علوم",
    r"رياضيات|math": "رياضيات",
    r"لغة\s*عربية|عربي(?!\s*و)|arabic": "لغة عربية",
    r"لغة\s*إنجليزية|انجليزي|english": "لغة إنجليزية",
    r"تاريخ|history": "تاريخ",
    r"جغرافيا|geography": "جغرافيا",
    r"دراسات\s*اجتماعية|اجتماعيات|social": "دراسات اجتماعية",
    r"فيزياء|physics": "فيزياء",
    r"كيمياء|chemistry": "كيمياء",
    r"أحياء|biology": "أحياء",
    r"دين|تربية\s*دينية|religion": "تربية دينية",
}

TERM_PATTERNS: dict[str, str] = {
    r"الترم\s*الأول|الفصل\s*الأول|term\s*1|first\s*term": "الأول",
    r"الترم\s*الثاني|الفصل\s*الثاني|term\s*2|second\s*term": "الثاني",
}

DOMAIN_PATTERNS: dict[str, str] = {
    r"منهج|كتاب|مقرر|curriculum|book": "curriculum",
    r"امتحان|اختبار|درجات|exam|test": "exams",
    r"مدرسة|نتيجة|طالب|school|student": "schools",
    r"أخبار|قرار|تحديث|news|update": "news",
}

# Subject → settings.py sources category
SUBJECT_TO_CATEGORY: dict[str, str] = {
    "رياضيات": "math",
    "علوم": "science",
    "فيزياء": "science",
    "كيمياء": "science",
    "أحياء": "science",
    "لغة عربية": "arabic",
    "تاريخ": "history",
    "جغرافيا": "history",
    "دراسات اجتماعية": "history",
    "لغة إنجليزية": "curriculum",
    "تربية دينية": "curriculum",
}


@dataclass
class QueryMetadata:
    raw_question: str
    domain: str = "curriculum"
    grade: str | None = None
    subject: str | None = None
    term: str | None = None
    source_category: str = "curriculum"
    keywords: list[str] = field(default_factory=list)
    search_query: str = ""
    needs_live_search: bool = False


def analyze_query(question: str) -> QueryMetadata:
    meta = QueryMetadata(raw_question=question)
    # Use original text for Arabic patterns (Arabic has no case)
    text = question

    for pattern, value in GRADE_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            meta.grade = value
            break

    for pattern, value in SUBJECT_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            meta.subject = value
            break

    for pattern, value in TERM_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            meta.term = value
            break

    for pattern, value in DOMAIN_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            meta.domain = value
            break

    # Map subject → source category
    if meta.subject:
        meta.source_category = SUBJECT_TO_CATEGORY.get(meta.subject, "curriculum")
    else:
        meta.source_category = meta.domain

    live_triggers = [
        r"أخبار", r"جديد", r"تحديث", r"الآن", r"هذا\s*العام",
        r"news", r"latest", r"update", r"2024", r"2025", r"2026",
    ]
    meta.needs_live_search = any(
        re.search(p, text, re.IGNORECASE) for p in live_triggers
    )

    parts = [p for p in [meta.subject, meta.grade, meta.term, "مصر", "وزارة التربية والتعليم"] if p]
    meta.search_query = " ".join(parts)
    meta.keywords = [p for p in [meta.subject, meta.grade, meta.term] if p]

    logger.info(
        f"Query analyzed: grade={meta.grade}, subject={meta.subject}, "
        f"term={meta.term}, domain={meta.domain}, live={meta.needs_live_search}"
    )
    return meta
