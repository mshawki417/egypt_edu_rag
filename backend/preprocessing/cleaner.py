"""
backend/preprocessing/cleaner.py
 
Arabic-aware text cleaning and normalization pipeline.
"""
from __future__ import annotations
import re
import unicodedata
 
 
# ── Arabic normalization helpers ───────────────────────────────────────────────
_ALEF_VARIANTS = re.compile(r"[أإآٱ]")
_TAMARBUTAH    = re.compile(r"ة\b")
_DIACRITICS    = re.compile(r"[\u064B-\u065F\u0670]")   # Harakat
_PUNCTUATION   = re.compile(r"[،؛؟!٪]")
_EXTRA_SPACES  = re.compile(r"[ \t]+")
_EMPTY_LINES   = re.compile(r"\n{3,}")
_PAGE_MARKERS  = re.compile(r"\[صفحة \d+\]")
 
# ✅ جديد: أنماط لإزالة محتوى Google UI الإنجليزي
_GOOGLE_UI     = re.compile(
    r"(Sign in|Google Search|Search tools|SafeSearch|Settings|"
    r"About \d+ results|Did you mean|All results|Images|Videos|News|Maps|"
    r"Web history|Search settings|Advanced search)",
    re.IGNORECASE,
)
 
 
def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text for consistent matching:
    - Unify Alef variants → ا
    - Remove diacritics (tashkeel / harakat)
    - Normalize Teh Marbuta
    - Collapse whitespace
    """
    text = unicodedata.normalize("NFC", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _DIACRITICS.sub("", text)
    text = _TAMARBUTAH.sub("ه", text)
    return text
 
 
def clean_text(text: str, normalize: bool = True) -> str:
    """Full cleaning pipeline for a raw scraped text block."""
    # Remove HTML artifacts that slipped through
    text = re.sub(r"<[^>]+>", " ", text)
    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)
    # Remove page markers from PDF extraction
    text = _PAGE_MARKERS.sub("", text)
    # ✅ إزالة عبارات Google UI الإنجليزية
    text = _GOOGLE_UI.sub(" ", text)
    # Normalize Arabic
    if normalize:
        text = normalize_arabic(text)
    # Collapse whitespace
    text = _EXTRA_SPACES.sub(" ", text)
    text = _EMPTY_LINES.sub("\n\n", text)
    return text.strip()
 
 
def is_arabic_heavy(text: str, threshold: float = 0.3) -> bool:
    """Return True if at least `threshold` fraction of chars are Arabic."""
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return arabic_chars / max(len(text), 1) >= threshold
 
 
# ✅ دالة جديدة: استخراج الأجزاء العربية فقط من نص مختلط
def extract_arabic_sections(text: str, min_section_len: int = 50) -> str:
    """
    من نص مختلط عربي/إنجليزي، بيستخرج الأجزاء العربية فقط.
    مفيد لصفحات Google اللي فيها نتائج عربية وسط UI إنجليزي.
    """
    lines = text.split("\n")
    arabic_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        arabic_chars = sum(1 for c in line if "\u0600" <= c <= "\u06FF")
        ratio = arabic_chars / max(len(line), 1)
        # ✅ خد أي سطر فيه عربي أكتر من 20% وطوله معقول
        if ratio >= 0.20 and len(line) >= min_section_len:
            arabic_lines.append(line)
    return "\n\n".join(arabic_lines)
 
