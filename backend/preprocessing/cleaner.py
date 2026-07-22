"""
backend/preprocessing/cleaner.py
Arabic-aware text cleaning and normalization pipeline.
"""
from __future__ import annotations
import re
import unicodedata

_ALEF_VARIANTS = re.compile(r"[أإآٱ]")
_TAMARBUTAH    = re.compile(r"ة(?=\s|$)")   # fixed: \b doesn't work with Arabic
_DIACRITICS    = re.compile(r"[\u064B-\u065F\u0670]")
_EXTRA_SPACES  = re.compile(r"[ \t]+")
_EMPTY_LINES   = re.compile(r"\n{3,}")
_PAGE_MARKERS  = re.compile(r"\[صفحة \d+\]")
_HTML_TAGS     = re.compile(r"<[^>]+>")
_URLS          = re.compile(r"https?://\S+")


def normalize_arabic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _ALEF_VARIANTS.sub("ا", text)
    text = _DIACRITICS.sub("", text)
    text = _TAMARBUTAH.sub("ه", text)
    return text


def clean_text(text: str, normalize: bool = True) -> str:
    text = _HTML_TAGS.sub(" ", text)
    text = _URLS.sub(" ", text)
    text = _PAGE_MARKERS.sub("", text)
    if normalize:
        text = normalize_arabic(text)
    text = _EXTRA_SPACES.sub(" ", text)
    text = _EMPTY_LINES.sub("\n\n", text)
    return text.strip()


def is_arabic_heavy(text: str, threshold: float = 0.15) -> bool:
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    return arabic_chars / max(len(text), 1) >= threshold


def extract_arabic_sections(text: str, min_len: int = 40) -> str:
    lines = text.split("\n")
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        arabic = sum(1 for c in line if "\u0600" <= c <= "\u06FF")
        if arabic / max(len(line), 1) >= 0.20 and len(line) >= min_len:
            out.append(line)
    return "\n\n".join(out)
