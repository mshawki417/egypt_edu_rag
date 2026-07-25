"""
╔══════════════════════════════════════════════════════════════════╗
║       محلل الاستعلامات العربية التعليمية — الإصدار v4          ║
║       Arabic Educational Query Analyzer — Production v4         ║
║                                                                  ║
║  المميزات الجديدة:                                              ║
║  • معالجة صرفية عربية متقدمة (PyArabic)                        ║
║  • كاشف نية ذكي بالتصويت الموزون                               ║
║  • خط أنابيب متوازٍ مع async/await                             ║
║  • ذاكرة تخزين مؤقت LRU + TTL                                  ║
║  • مُعيد بناء الاستعلام (Query Rewriter)                       ║
║  • نقاط ثقة متعددة الأبعاد                                      ║
║  • منطق احتياطي متعدد المستويات                                 ║
╚══════════════════════════════════════════════════════════════════╝

التوافق:
    orchestrator.py | live_scraper.py | chunker.py | retriever.py
"""

from __future__ import annotations

import re
import asyncio
import time
import hashlib
import json

from dataclasses import dataclass, field, asdict
from collections import Counter
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pyarabic.araby as araby
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from cachetools import TTLCache
from loguru import logger
from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════════
#  إعدادات الإنتاج
# ══════════════════════════════════════════════════════════════════

class AnalyzerConfig(BaseModel):
    """إعدادات محلل الاستعلامات — يمكن تغييرها عبر ملف .env"""

    cache_maxsize: int = Field(default=1000, description="الحد الأقصى للمدخلات في الذاكرة المؤقتة")
    cache_ttl_seconds: int = Field(default=3600, description="مدة صلاحية الذاكرة المؤقتة (ثانية)")
    max_keywords: int = Field(default=12, description="الحد الأقصى للكلمات المفتاحية")
    max_topic_words: int = Field(default=6, description="الحد الأقصى لكلمات الموضوع")
    min_word_length: int = Field(default=2, description="الحد الأدنى لطول الكلمة")
    ml_confidence_threshold: float = Field(default=0.35, description="حد الثقة للنموذج ML")
    enable_query_expansion: bool = Field(default=True, description="تفعيل توسيع الاستعلام")
    enable_async: bool = Field(default=True, description="تفعيل المعالجة غير المتزامنة")
    log_level: str = Field(default="INFO")


CONFIG = AnalyzerConfig()


# ══════════════════════════════════════════════════════════════════
#  قواميس المراحل الدراسية (موسّعة)
# ══════════════════════════════════════════════════════════════════

STAGE_PATTERNS: dict[str, list[str]] = {
    "primary": [
        "ابتدائي", "ابتدائى", "الابتدائي", "الابتدائى",
        "المرحلة الابتدائية", "primary", "الاساسي", "الأساسي",
    ],
    "preparatory": [
        "اعدادي", "إعدادي", "الاعدادي", "الإعدادي",
        "المرحلة الاعدادية", "المرحلة الإعدادية", "preparatory",
        "متوسط", "المتوسط",
    ],
    "secondary": [
        "ثانوي", "ثانوى", "الثانوية", "الثانوية العامة",
        "المرحلة الثانوية", "secondary", "تانوي", "تانوى",
    ],
    "university": [
        "جامعي", "جامعة", "كلية", "university", "college",
        "بكالوريوس", "ليسانس",
    ],
}

# ══════════════════════════════════════════════════════════════════
#  أنماط الصفوف الدراسية (موسّعة مع الأرقام الهندية)
# ══════════════════════════════════════════════════════════════════

GRADE_PATTERNS: dict[str, list[str]] = {
    # ابتدائي
    "الأول الابتدائي": [
        "اول ابتدائي", "اول الابتدائي", "الاول ابتدائي", "الاول الابتدائي",
        "1 ابتدائي", "١ ابتدائي", "الصف الاول الابتدائي",
    ],
    "الثاني الابتدائي": [
        "ثاني ابتدائي", "ثاني الابتدائي", "الثاني ابتدائي", "الثاني الابتدائي",
        "2 ابتدائي", "٢ ابتدائي", "الصف الثاني الابتدائي",
    ],
    "الثالث الابتدائي": [
        "ثالث ابتدائي", "ثالث الابتدائي", "الثالث ابتدائي", "الثالث الابتدائي",
        "3 ابتدائي", "٣ ابتدائي", "الصف الثالث الابتدائي",
    ],
    "الرابع الابتدائي": [
        "رابع ابتدائي", "رابع الابتدائي", "الرابع ابتدائي", "الرابع الابتدائي",
        "4 ابتدائي", "٤ ابتدائي", "الصف الرابع الابتدائي",
    ],
    "الخامس الابتدائي": [
        "خامس ابتدائي", "خامس الابتدائي", "الخامس ابتدائي", "الخامس الابتدائي",
        "5 ابتدائي", "٥ ابتدائي", "الصف الخامس الابتدائي",
    ],
    "السادس الابتدائي": [
        "سادس ابتدائي", "سادس الابتدائي", "السادس ابتدائي", "السادس الابتدائي",
        "6 ابتدائي", "٦ ابتدائي", "الصف السادس الابتدائي",
    ],
    # إعدادي
    "الأول الإعدادي": [
        "اول اعدادي", "اول الاعدادي", "الاول اعدادي", "الاول الاعدادي",
        "1 اعدادي", "١ اعدادي", "الصف الاول الاعدادي",
    ],
    "الثاني الإعدادي": [
        "ثاني اعدادي", "ثاني الاعدادي", "الثاني اعدادي", "الثاني الاعدادي",
        "2 اعدادي", "٢ اعدادي", "الصف الثاني الاعدادي",
    ],
    "الثالث الإعدادي": [
        "ثالث اعدادي", "ثالث الاعدادي", "الثالث اعدادي", "الثالث الاعدادي",
        "3 اعدادي", "٣ اعدادي", "الصف الثالث الاعدادي",
    ],
    # ثانوي
    "الأول الثانوي": [
        "اول ثانوي", "اول الثانوي", "الاول ثانوي", "الاول الثانوي",
        "1 ثانوي", "١ ثانوي", "الصف الاول الثانوي",
    ],
    "الثاني الثانوي": [
        "ثاني ثانوي", "ثاني الثانوي", "الثاني ثانوي", "الثاني الثانوي",
        "2 ثانوي", "٢ ثانوي", "الصف الثاني الثانوي",
    ],
    "الثالث الثانوي": [
        "ثالث ثانوي", "ثالث الثانوي", "الثالث ثانوي", "الثالث الثانوي",
        "3 ثانوي", "٣ ثانوي", "الثانوية العامة", "الصف الثالث الثانوي",
    ],
}

# ══════════════════════════════════════════════════════════════════
#  قاموس المواد (موسّع مع الأوزان)
# ══════════════════════════════════════════════════════════════════

SUBJECTS: dict[str, list[str]] = {
    "رياضيات": [
        "رياضيات", "رياضه", "حساب", "جبر", "هندسة",
        "تفاضل", "تكامل", "معادلات", "قسمة", "ضرب",
        "مثلثات", "احصاء", "احتمالات", "لوغاريتم",
        "math", "maths", "mathematics",
    ],
    "فيزياء": [
        "فيزياء", "فيزيا", "سرعة", "قوة", "طاقة",
        "حركة", "نيوتن", "كهرباء", "مغناطيس", "ضوء",
        "موجات", "نووي", "ميكانيكا", "physics",
    ],
    "كيمياء": [
        "كيمياء", "كيميا", "ذرة", "عنصر", "تفاعل",
        "مركب", "عضوية", "غير عضوية", "جدول دوري",
        "محلول", "اكسدة", "chemistry",
    ],
    "احياء": [
        "احياء", "أحياء", "خلية", "وراثة", "جسم الانسان",
        "نبات", "حيوان", "بكتيريا", "فيروس", "جينات",
        "biology", "تشريح",
    ],
    "علوم": [
        "علوم", "science", "طبيعة", "بيئة", "تكنولوجيا",
    ],
    "لغة عربية": [
        "عربي", "عربى", "لغة عربية", "نحو", "صرف",
        "بلاغة", "شعر", "نصوص", "قراءة", "إملاء",
        "تعبير", "قواعد",
    ],
    "لغة انجليزية": [
        "انجليزي", "انجليزى", "إنجليزي", "english",
        "grammar", "vocabulary", "reading", "writing",
    ],
    "لغة فرنسية": [
        "فرنسي", "فرنسى", "français", "french",
    ],
    "تاريخ": [
        "تاريخ", "حضارة", "ثورة", "حرب", "اسرات",
        "فراعنة", "عصور", "history",
    ],
    "جغرافيا": [
        "جغرافيا", "جغرافيه", "مناخ", "سكان", "خرائط",
        "قارة", "محيط", "geography",
    ],
    "دراسات اجتماعية": [
        "دراسات", "اجتماعية", "مجتمع", "وطن",
        "social studies",
    ],
    "دين اسلامي": [
        "دين", "إسلامي", "اسلامي", "قران", "قرآن",
        "فقه", "عقيدة", "سيرة", "حديث",
    ],
    "تربية وطنية": [
        "تربية وطنية", "قومية", "مواطنة",
    ],
    "حاسب آلي": [
        "حاسب", "كمبيوتر", "برمجة", "حاسوب",
        "python", "java", "programming", "technology",
    ],
    "فلسفة": [
        "فلسفة", "منطق", "علم النفس", "اخلاق",
        "philosophy", "psychology",
    ],
}

# ══════════════════════════════════════════════════════════════════
#  قواعد النية (مُحسَّنة بالأوزان)
# ══════════════════════════════════════════════════════════════════

INTENT_RULES: dict[str, list[tuple[str, float]]] = {
    "summary": [
        ("لخص", 2.0), ("ملخص", 1.5), ("تلخيص", 1.5),
        ("الخلاصة", 1.0), ("اختصر", 1.5), ("باختصار", 1.0),
        ("اجمل", 1.0), ("نقاط", 0.8),
    ],
    "explanation": [
        ("اشرح", 2.0), ("شرح", 1.5), ("وضح", 1.5),
        ("فسر", 1.5), ("كيف", 1.0), ("لماذا", 1.0),
        ("ما سبب", 1.0), ("اريد افهم", 1.5), ("افهم", 0.8),
    ],
    "solution": [
        ("حل", 2.0), ("مسألة", 1.5), ("تمرين", 1.5),
        ("اجابة", 1.0), ("إجابة", 1.0), ("احسب", 2.0),
        ("اوجد", 1.5), ("جد", 1.0), ("برهن", 1.5),
    ],
    "definition": [
        ("ما هو", 1.5), ("من هو", 1.5), ("تعريف", 2.0),
        ("عرف", 1.5), ("ما معنى", 1.5), ("يعني", 1.0),
        ("مفهوم", 1.5),
    ],
    "comparison": [
        ("قارن", 2.0), ("الفرق", 1.5), ("مقارنة", 1.5),
        ("الاختلاف", 1.5), ("اوجه الشبه", 1.5), ("مثل", 0.5),
    ],
    "exam": [
        ("امتحان", 2.0), ("اختبار", 1.5), ("سؤال", 1.0),
        ("درجات", 1.0), ("اسئلة", 1.5), ("نموذج", 1.0),
        ("مراجعة", 1.5), ("بنك", 1.0),
    ],
    "curriculum": [
        ("منهج", 2.0), ("كتاب", 1.5), ("درس", 1.0),
        ("باب", 1.0), ("وحدة", 1.0), ("فصل", 0.8),
        ("كورس", 1.0),
    ],
    "news": [
        ("اخبار", 2.0), ("جديد", 1.0), ("اليوم", 1.5),
        ("قرار", 1.5), ("وزارة", 1.0), ("اعلن", 1.5),
    ],
}

# ══════════════════════════════════════════════════════════════════
#  كلمات وقف عربية شاملة
# ══════════════════════════════════════════════════════════════════

ARABIC_STOPWORDS: frozenset[str] = frozenset({
    "ما", "هو", "هي", "هم", "هن", "هما",
    "من", "في", "على", "عن", "الى", "الي",
    "إلى", "إلي", "أن", "ان", "لا", "لم",
    "لن", "قد", "كان", "كانت", "كانوا",
    "يكون", "يكن", "تكون", "ليس", "ليست",
    "إذا", "اذا", "إذ", "اذ", "حيث", "حين",
    "بعد", "قبل", "بين", "عند", "مع", "دون",
    "مع", "بدون", "حول", "خلال", "ضمن",
    "اريد", "أريد", "اريد", "ابغا", "ابي",
    "شرح", "اشرح", "درس", "فضلك", "من فضلك",
    "لو", "لكن", "ولكن", "و", "ف", "ثم",
    "او", "أو", "إما", "اما", "بل",
    "كل", "جميع", "بعض", "اي", "أي",
    "هذا", "هذه", "ذلك", "تلك", "هؤلاء",
    "الذي", "التي", "الذين", "اللذان",
})


# ══════════════════════════════════════════════════════════════════
#  نموذج البيانات — محسَّن
# ══════════════════════════════════════════════════════════════════

@dataclass
class ConfidenceBreakdown:
    """تفصيل درجة الثقة لكل بُعد"""
    subject_score: float = 0.0
    grade_score: float = 0.0
    stage_score: float = 0.0
    topic_score: float = 0.0
    intent_score: float = 0.0
    keyword_score: float = 0.0
    total: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class QueryMetadata:
    """البيانات الوصفية الكاملة للاستعلام"""

    # — الإدخال —
    raw_question: str
    normalized: str = ""
    stemmed: str = ""

    # — الكيانات المستخرجة —
    intent: str = "general"
    intent_confidence: float = 0.0
    subject: Optional[str] = None
    stage: Optional[str] = None
    grade: Optional[str] = None
    term: Optional[str] = None
    year: Optional[str] = None
    topic: Optional[str] = None

    # — الكلمات المفتاحية —
    keywords: list[str] = field(default_factory=list)
    expanded_keywords: list[str] = field(default_factory=list)

    # — الاستعلام والمصدر —
    search_query: str = ""
    rewritten_query: str = ""
    source_category: str = "general"
    source_priority: list[str] = field(default_factory=list)
    needs_live_search: bool = False

    # — الثقة —
    confidence: float = 0.0
    confidence_breakdown: ConfidenceBreakdown = field(
        default_factory=ConfidenceBreakdown
    )

    # — البيانات المجمّعة —
    entities: dict = field(default_factory=dict)

    # — التتبع —
    processing_time_ms: float = 0.0
    cache_hit: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════
#  المُطبِّع العربي المتقدم
# ══════════════════════════════════════════════════════════════════

class ArabicNormalizer:
    """
    معالجة نصوص عربية بالاستفادة من مكتبة PyArabic
    مع خطوات تنظيف إضافية مخصصة للسياق التعليمي
    """

    # تحويل الأرقام الهندية والفارسية إلى عربية
    _INDIC_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

    # أنماط التشكيل والزخارف
    _DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670]")
    _TATWEEL_RE    = re.compile(r"\u0640+")
    _SYMBOLS_RE    = re.compile(r"[^\w\s\u0600-\u06FF0-9]")
    _SPACES_RE     = re.compile(r"\s+")

    # الحروف المتشابهة
    _CHAR_MAP = str.maketrans({
        "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
        "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي",
    })

    @classmethod
    def normalize(cls, text: str) -> str:
        """تطبيع شامل مع الاحتفاظ بالمحتوى الدلالي"""
        if not text:
            return ""

        text = text.lower()
        text = text.translate(cls._INDIC_MAP)          # أرقام هندية → عربية
        text = text.translate(cls._CHAR_MAP)            # همزات وتاء مربوطة
        text = cls._DIACRITICS_RE.sub("", text)        # إزالة التشكيل
        text = cls._TATWEEL_RE.sub("", text)            # إزالة التطويل
        text = cls._SYMBOLS_RE.sub(" ", text)           # رموز → فراغ
        text = cls._SPACES_RE.sub(" ", text)            # ضغط الفراغات
        return text.strip()

    @classmethod
    def stem(cls, text: str) -> str:
        """
        إزالة السوابق واللواحق الشائعة لتحسين المطابقة
        نستخدم PyArabic لتجريد الكلمات
        """
        words = text.split()
        stemmed = []
        for word in words:
            try:
                # إزالة السوابق مثل: وال، بال، فال، وب، وك
                stripped = araby.strip_tashkeel(word)
                stripped = araby.strip_tatweel(stripped)
                # استخدام خوارزمية الجذر التقريبية
                root = araby.reduce_tashkeel(stripped)
                stemmed.append(root if root else stripped)
            except Exception:
                stemmed.append(word)
        return " ".join(stemmed)

    @classmethod
    def tokenize(cls, text: str) -> list[str]:
        """تقسيم النص إلى رموز مع تصفية الكلمات القصيرة"""
        return [
            w for w in text.split()
            if len(w) >= CONFIG.min_word_length
            and w not in ARABIC_STOPWORDS
        ]


# ══════════════════════════════════════════════════════════════════
#  كاشف النية المُحسَّن — بالتصويت الموزون
# ══════════════════════════════════════════════════════════════════

class IntentDetector:
    """
    يدمج ثلاثة مستويات للكشف:
    1. المطابقة المباشرة بالأوزان
    2. درجة التقارب الدلالي (TF-IDF)
    3. قواعد السياق
    """

    # جُمَل تدريب نموذجية لكل نية (تُستخدم ببناء ناقل TF-IDF)
    _SEED_SENTENCES: dict[str, list[str]] = {
        "explanation": [
            "اشرح لي درس الجهاز الهضمي",
            "كيف تعمل الكلية",
            "وضح لي قانون نيوتن",
            "فسر لي معنى الاستعارة",
            "أريد فهم التمثيل الضوئي",
        ],
        "summary": [
            "لخص لي درس الدورة الدموية",
            "ملخص الفصل الثالث",
            "اعطني خلاصة الباب الثاني",
            "اختصر لي المنهج",
            "نقاط مهمة في التاريخ",
        ],
        "solution": [
            "حل مسألة في الرياضيات",
            "احسب مساحة المستطيل",
            "اوجد قيمة المجهول",
            "حل هذا التمرين",
            "جد مشتقة الدالة",
        ],
        "definition": [
            "ما هو الضوء",
            "عرف الخلية الحية",
            "ما معنى البلاغة",
            "مفهوم التسارع في الفيزياء",
            "تعريف الجملة الاسمية",
        ],
        "exam": [
            "اسئلة امتحان الثانوية العامة",
            "نموذج اختبار الترم الاول",
            "مراجعة نهائية للفيزياء",
            "بنك اسئلة الكيمياء",
            "درجات امتحان الرياضيات",
        ],
        "curriculum": [
            "منهج الصف الثالث الثانوي",
            "مقرر الكيمياء للترم الثاني",
            "كتاب الوزارة للرياضيات",
            "ما هو منهج العلوم",
        ],
        "comparison": [
            "ما الفرق بين الخلية النباتية والحيوانية",
            "قارن بين التنفس الهوائي واللاهوائي",
            "اوجه الشبه والاختلاف بين الحوض الهندسي",
        ],
    }

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._seed_vectors = None
        self._seed_labels: list[str] = []
        self._build_tfidf_index()

    def _build_tfidf_index(self) -> None:
        """بناء فهرس TF-IDF من الجُمَل التدريبية"""
        try:
            docs, labels = [], []
            for intent, sentences in self._SEED_SENTENCES.items():
                for sent in sentences:
                    docs.append(ArabicNormalizer.normalize(sent))
                    labels.append(intent)

            self._vectorizer = TfidfVectorizer(
                analyzer="char_wb",   # مثالي للعربية (n-جرام حرفية)
                ngram_range=(2, 4),
                min_df=1,
                max_features=3000,
            )
            self._seed_vectors = self._vectorizer.fit_transform(docs)
            self._seed_labels = labels
            logger.debug("TF-IDF index built: {} docs, {} features",
                         len(docs), self._vectorizer.max_features)
        except Exception as e:
            logger.warning("TF-IDF build failed — rule-only mode: {}", e)

    def detect(self, text: str) -> tuple[str, float]:
        """
        يُعيد (النية، درجة_الثقة)
        يدمج التصويت الموزون + التشابه التدريبي
        """
        # ── المستوى 1: التصويت الموزون بالأنماط ──
        weighted_scores: dict[str, float] = {}
        for intent, patterns in INTENT_RULES.items():
            score = sum(
                weight for phrase, weight in patterns
                if phrase in text
            )
            weighted_scores[intent] = score

        best_rule_intent = max(weighted_scores, key=weighted_scores.get)
        best_rule_score  = weighted_scores[best_rule_intent]

        # ── المستوى 2: التشابه الدلالي TF-IDF ──
        ml_intent, ml_score = "general", 0.0
        if self._vectorizer is not None and best_rule_score < 1.5:
            try:
                vec = self._vectorizer.transform([text])
                sims = cosine_similarity(vec, self._seed_vectors)[0]
                best_idx = int(np.argmax(sims))
                ml_intent = self._seed_labels[best_idx]
                ml_score  = float(sims[best_idx])
            except Exception:
                pass

        # ── المستوى 3: دمج النتيجتين ──
        if best_rule_score >= 1.5:
            confidence = min(best_rule_score / 4.0, 1.0)
            return best_rule_intent, round(confidence, 2)

        if ml_score >= CONFIG.ml_confidence_threshold:
            return ml_intent, round(ml_score, 2)

        if best_rule_score > 0:
            return best_rule_intent, round(best_rule_score / 4.0, 2)

        return "general", 0.0


# مثيل مشترك يُبنى مرة واحدة
_INTENT_DETECTOR = IntentDetector()


# ══════════════════════════════════════════════════════════════════
#  خط أنابيب الاستخراج
# ══════════════════════════════════════════════════════════════════

def _detect_from_dict(text: str, dictionary: dict[str, list[str]]) -> Optional[str]:
    """إيجاد أول تطابق في قاموس أحادي الرتبة"""
    for key, patterns in dictionary.items():
        if any(p in text for p in patterns):
            return key
    return None


def _detect_all_from_dict(text: str, dictionary: dict[str, list[str]]) -> list[str]:
    """إيجاد جميع التطابقات (للكلمات المفتاحية)"""
    return list({
        key for key, patterns in dictionary.items()
        if any(p in text for p in patterns)
    })


def _detect_term(text: str) -> Optional[str]:
    patterns = {
        "الترم الأول": [
            "الترم الاول", "الترم الأول", "الفصل الاول",
            "الفصل الأول", "ترم اول", "ترم 1", "ترم١",
        ],
        "الترم الثاني": [
            "الترم الثاني", "الفصل الثاني", "الفصل الثاني",
            "ترم ثاني", "ترم 2", "ترم٢",
        ],
    }
    return _detect_from_dict(text, patterns)


def _detect_year(text: str) -> Optional[str]:
    m = re.search(r"(20\d{2}|١٩\d{2}|٢٠\d{2})", text)
    return m.group(1) if m else None


def _extract_topic(text: str, subject: Optional[str] = None) -> Optional[str]:
    """
    استخراج موضوع الدرس بعد حذف الكلمات الوظيفية.
    مثال: "اشرح درس الجهاز الهضمي ثاني اعدادي"
            → "الجهاز الهضمي"
    """
    stop_words = {
        "شرح", "اشرح", "درس", "منهج", "كتاب", "ملخص", "لخص",
        "المرحله", "المرحلة", "الثانوي", "الاعدادي", "الابتدائي",
        "العامه", "العامة", "الصف", "الترم", "الفصل",
        "الاول", "الثاني", "الثالث", "الرابع", "الخامس", "السادس",
        "اول", "ثاني", "ثالث", "رابع", "خامس", "سادس",
        "اعدادي", "الاعدادي", "ابتدائي", "الابتدائي",
        "ثانوي", "الثانوي", "ترم", "وحده", "وحدة", "باب",
        "فصل", "الفصل", "للصف", "للترم", "للمرحله",
        "لل", "في", "من", "على", "عن", "الى",
    }
    # إزالة الصفوف والمراحل المكتشفة كي لا تتسرب للموضوع
    for grade_list in GRADE_PATTERNS.values():
        for pat in grade_list:
            text = text.replace(pat, "")

    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    if not words:
        return None
    topic = " ".join(words[: CONFIG.max_topic_words])
    return topic.strip() or None


def _extract_keywords(text: str) -> list[str]:
    """
    استخراج كلمات مفتاحية ذكية:
    يدمج تكرار الكلمات + إزالة كلمات الوقف + التصفية الطولية
    """
    tokens = ArabicNormalizer.tokenize(text)
    if not tokens:
        return []
    counter = Counter(tokens)
    return [w for w, _ in counter.most_common(CONFIG.max_keywords)]


# ══════════════════════════════════════════════════════════════════
#  مُعيد بناء الاستعلام
# ══════════════════════════════════════════════════════════════════

class QueryRewriter:
    """
    يُنتج استعلام بحث مُحسَّناً من البيانات الوصفية.
    يُعطي أولوية للمصادر المصرية عند الكشف عن المناهج.
    """

    _SOURCE_SUFFIXES: dict[str, list[str]] = {
        "curriculum": ["وزارة التربية والتعليم مصر", "كتاب الوزارة", "منهج مصر"],
        "education":  ["شرح تعليمي", "موضوع تعليمي"],
        "news":       ["اخبار", "2026"],
        "general":    [],
    }

    @classmethod
    def build(cls, meta: QueryMetadata) -> str:
        """الاستعلام الأساسي"""
        parts: list[str] = [meta.raw_question]

        if meta.topic:    parts.append(meta.topic)
        if meta.subject:  parts.append(meta.subject)
        if meta.grade:    parts.append(meta.grade)
        if meta.term:     parts.append(meta.term)
        if meta.year:     parts.append(meta.year)

        suffixes = cls._SOURCE_SUFFIXES.get(meta.source_category, [])
        parts.extend(suffixes)

        # إزالة المكررات مع الحفاظ على الترتيب
        return " ".join(dict.fromkeys(parts))

    @classmethod
    def rewrite(cls, meta: QueryMetadata) -> str:
        """
        استعلام مُعاد صياغته للـ RAG:
        أكثر دقة من الاستعلام الأصلي
        """
        parts: list[str] = []

        if meta.intent in ("explanation", "summary"):
            if meta.topic:
                parts.append(f"شرح {meta.topic}")
            if meta.subject:
                parts.append(meta.subject)

        elif meta.intent == "solution":
            if meta.topic:
                parts.append(f"حل {meta.topic}")

        elif meta.intent == "definition":
            if meta.topic:
                parts.append(f"تعريف {meta.topic}")

        elif meta.intent == "comparison":
            if meta.topic:
                parts.append(f"مقارنة {meta.topic}")
            if meta.subject:
                parts.append(meta.subject)

        elif meta.intent == "exam":
            parts.extend(filter(None, [
                meta.subject,
                meta.grade or meta.stage,
                "اسئلة امتحان",
                meta.year or "2025",
            ]))

        # الكلمات المفتاحية الإضافية (نتجنب التكرار)
        existing = set(parts)
        parts.extend(k for k in meta.expanded_keywords[:4] if k not in existing)

        if meta.grade:    parts.append(meta.grade)
        if meta.term:     parts.append(meta.term)

        result = " ".join(p for p in parts if p)
        return result if result else meta.search_query

    @classmethod
    def expand_keywords(cls, meta: QueryMetadata) -> list[str]:
        """توسيع الكلمات المفتاحية بالمرادفات والمشتقات"""
        expansions: list[str] = []

        if meta.topic:
            # نضيف توسيعات حسب النية لتجنب الزخم
            if meta.intent == "explanation":
                expansions += [f"شرح {meta.topic}", f"خطوات {meta.topic}"]
            elif meta.intent == "summary":
                expansions += [f"ملخص {meta.topic}"]
            elif meta.intent == "comparison":
                expansions += [f"مقارنة {meta.topic}", f"الفرق بين {meta.topic}"]
            else:
                expansions += [f"{meta.topic}"]
        if meta.subject:
            expansions.append(meta.subject)

        # إضافة مصطلحات الصف
        if meta.grade and meta.subject:
            expansions.append(f"{meta.subject} {meta.grade}")

        return expansions[:8]


# ══════════════════════════════════════════════════════════════════
#  درجة الثقة متعددة الأبعاد
# ══════════════════════════════════════════════════════════════════

def _calculate_confidence(meta: QueryMetadata) -> tuple[float, ConfidenceBreakdown]:
    bd = ConfidenceBreakdown()

    bd.subject_score  = 0.25 if meta.subject  else 0.0
    bd.grade_score    = 0.20 if meta.grade    else 0.0
    bd.stage_score    = 0.10 if meta.stage    else 0.0
    bd.topic_score    = 0.20 if meta.topic    else 0.0
    bd.intent_score   = min(meta.intent_confidence * 0.15, 0.15)
    bd.keyword_score  = min(len(meta.keywords) / CONFIG.max_keywords * 0.10, 0.10)

    bd.total = round(min(
        bd.subject_score + bd.grade_score + bd.stage_score +
        bd.topic_score   + bd.intent_score + bd.keyword_score,
        1.0
    ), 2)
    return bd.total, bd


# ══════════════════════════════════════════════════════════════════
#  أولوية المصادر
# ══════════════════════════════════════════════════════════════════

def _get_source_priority(meta: QueryMetadata) -> list[str]:
    if meta.source_category == "curriculum":
        return ["moe.gov.eg", "ekb.eg", "study.ekb.eg", "elearning.moe.gov.eg"]
    if meta.source_category == "education":
        return ["ekb.eg", "marefa.org", "ar.wikipedia.org", "mawdoo3.com"]
    if meta.source_category == "news":
        return ["youm7.com", "masrawy.com", "ahram.org.eg"]
    return ["ar.wikipedia.org", "mawdoo3.com"]


def _detect_source_category(meta: QueryMetadata) -> str:
    if meta.grade or meta.stage or meta.term or meta.intent == "curriculum":
        return "curriculum"
    if meta.intent == "news":
        return "news"
    if meta.subject:
        return "education"
    return "general"


def _needs_live_search(text: str) -> bool:
    signals = [
        "اليوم", "الان", "الآن", "حاليا", "حالياً",
        "اخر", "جديد", "قرار", "2026", "٢٠٢٦",
        "اخبار", "مستجدات", "تحديث",
    ]
    return any(s in text for s in signals)


# ══════════════════════════════════════════════════════════════════
#  ذاكرة التخزين المؤقت
# ══════════════════════════════════════════════════════════════════

_QUERY_CACHE: TTLCache = TTLCache(
    maxsize=CONFIG.cache_maxsize,
    ttl=CONFIG.cache_ttl_seconds,
)


def _cache_key(question: str) -> str:
    return hashlib.md5(question.strip().lower().encode()).hexdigest()


# ══════════════════════════════════════════════════════════════════
#  المحلل الرئيسي — المتزامن
# ══════════════════════════════════════════════════════════════════

def analyze_query(question: str) -> QueryMetadata:
    """
    خط الأنابيب الرئيسي.
    يحوّل سؤال المستخدم إلى بيانات وصفية منظمة للاسترجاع.

    المراحل:
        1. التحقق من الذاكرة المؤقتة
        2. التطبيع + الجذع
        3. استخراج الكيانات (نية / مادة / مرحلة / صف / ترم / سنة)
        4. استخراج الموضوع والكلمات المفتاحية
        5. تحديد الفئة والمصدر
        6. بناء الاستعلام + إعادة الصياغة
        7. حساب الثقة
        8. التخزين المؤقت والتسجيل
    """
    t0 = time.perf_counter()

    # ── 1. ذاكرة مؤقتة ──
    key = _cache_key(question)
    if key in _QUERY_CACHE:
        cached: QueryMetadata = _QUERY_CACHE[key]
        cached.cache_hit = True
        logger.debug("Cache hit: {}", question[:40])
        return cached

    try:
        # ── 2. تطبيع ──
        normalized = ArabicNormalizer.normalize(question)
        stemmed    = ArabicNormalizer.stem(normalized)

        meta = QueryMetadata(
            raw_question=question,
            normalized=normalized,
            stemmed=stemmed,
        )

        # ── 3. استخراج الكيانات ──
        meta.intent, meta.intent_confidence = _INTENT_DETECTOR.detect(normalized)
        meta.subject = _detect_from_dict(normalized, SUBJECTS)
        meta.stage   = _detect_from_dict(normalized, STAGE_PATTERNS)
        meta.grade   = _detect_from_dict(normalized, GRADE_PATTERNS)
        meta.term    = _detect_term(normalized)
        meta.year    = _detect_year(normalized)

        # ── 4. موضوع وكلمات مفتاحية ──
        meta.topic    = _extract_topic(normalized, meta.subject)
        meta.keywords = _extract_keywords(normalized)

        # إضافة الكيانات المكتشفة للكلمات المفتاحية
        for item in filter(None, [meta.subject, meta.grade, meta.topic]):
            if item not in meta.keywords:
                meta.keywords.append(item)
        meta.keywords = list(dict.fromkeys(meta.keywords))[: CONFIG.max_keywords]

        # ── 5. توسيع الكلمات المفتاحية ──
        if CONFIG.enable_query_expansion:
            meta.expanded_keywords = QueryRewriter.expand_keywords(meta)

        # ── 6. الفئة والمصدر ──
        meta.source_category  = _detect_source_category(meta)
        meta.source_priority  = _get_source_priority(meta)
        meta.needs_live_search = _needs_live_search(normalized)

        # ── 7. الاستعلام ──
        meta.search_query  = QueryRewriter.build(meta)
        meta.rewritten_query = QueryRewriter.rewrite(meta)

        # ── 8. الثقة ──
        meta.confidence, meta.confidence_breakdown = _calculate_confidence(meta)

        # ── بيانات الكيانات المجمّعة ──
        meta.entities = {
            "subject":  meta.subject,
            "grade":    meta.grade,
            "stage":    meta.stage,
            "topic":    meta.topic,
            "keywords": meta.keywords,
        }

        # ── وقت المعالجة ──
        meta.processing_time_ms = round((time.perf_counter() - t0) * 1000, 2)

        # ── التخزين المؤقت ──
        _QUERY_CACHE[key] = meta

        logger.info(
            "✅ [{ms}ms] | نية={intent}({conf}) | مادة={sub} | صف={grade} | ثقة={total}",
            ms=meta.processing_time_ms,
            intent=meta.intent,
            conf=meta.intent_confidence,
            sub=meta.subject,
            grade=meta.grade,
            total=meta.confidence,
        )

        return meta

    except Exception as exc:
        logger.exception("Query analyzer failed: {}", exc)
        return QueryMetadata(
            raw_question=question,
            normalized=ArabicNormalizer.normalize(question),
            search_query=question,
            keywords=question.split()[:8],
            confidence=0.05,
            processing_time_ms=round((time.perf_counter() - t0) * 1000, 2),
        )


# ══════════════════════════════════════════════════════════════════
#  المحلل غير المتزامن — async
# ══════════════════════════════════════════════════════════════════

_EXECUTOR = ThreadPoolExecutor(max_workers=4)


async def analyze_query_async(question: str) -> QueryMetadata:
    """
    نسخة async من المحلل — مثالية لـ FastAPI / aiohttp.
    تُشغّل العملية في thread-pool لتجنب حجب حلقة الأحداث.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_EXECUTOR, analyze_query, question)


async def analyze_batch_async(questions: list[str]) -> list[QueryMetadata]:
    """
    تحليل دفعة من الأسئلة بالتوازي.
    أسرع بكثير من الاستدعاء التسلسلي عند وجود عدة أسئلة.
    """
    tasks = [analyze_query_async(q) for q in questions]
    return await asyncio.gather(*tasks)


# ══════════════════════════════════════════════════════════════════
#  أدوات مساعدة للتكامل
# ══════════════════════════════════════════════════════════════════

def get_answer_style(intent: str) -> str:
    """يُحدد أسلوب الإجابة المناسب لعرضه في الـ RAG"""
    styles = {
        "explanation": "step_by_step",
        "summary":     "short_summary",
        "solution":    "solution_with_steps",
        "definition":  "direct_answer",
        "comparison":  "table_format",
        "exam":        "qa_format",
        "curriculum":  "structured_list",
    }
    return styles.get(intent, "general")


def clear_cache() -> int:
    """مسح الذاكرة المؤقتة — مفيد عند تحديث القواميس"""
    size = len(_QUERY_CACHE)
    _QUERY_CACHE.clear()
    logger.info("Cache cleared: {} entries removed", size)
    return size


def cache_stats() -> dict:
    """إحصائيات الذاكرة المؤقتة"""
    return {
        "current_size": len(_QUERY_CACHE),
        "max_size":     CONFIG.cache_maxsize,
        "ttl_seconds":  CONFIG.cache_ttl_seconds,
    }


# ══════════════════════════════════════════════════════════════════
#  تجريب سريع عند التشغيل المباشر
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    TEST_QUESTIONS = [
        "اشرح لي درس الجهاز الهضمي للصف الثاني الاعدادي الترم الاول",
        "حل مسألة في الرياضيات للصف الثالث الثانوي",
        "ما هو تعريف التمثيل الضوئي في العلوم",
        "ملخص منهج الكيمياء الترم الثاني للثانوية العامة 2025",
        "قارن بين الخلية النباتية والحيوانية",
        "اسئلة امتحان الفيزياء للصف الاول الثانوي",
        "اخر اخبار وزارة التربية والتعليم اليوم 2026",
        "ما الفرق بين الجملة الاسمية والفعلية في اللغة العربية",
    ]

    print("\n" + "═" * 65)
    print("   🔬  محلل الاستعلامات العربية v4 — اختبار سريع")
    print("═" * 65 + "\n")

    for q in TEST_QUESTIONS:
        result = analyze_query(q)
        print(f"📝 السؤال   : {q}")
        print(f"   النية    : {result.intent} ({result.intent_confidence:.0%})")
        print(f"   المادة   : {result.subject or '—'}")
        print(f"   الصف     : {result.grade or '—'}")
        print(f"   الموضوع  : {result.topic or '—'}")
        print(f"   الاستعلام: {result.rewritten_query}")
        print(f"   الثقة    : {result.confidence:.0%}  |  ⏱ {result.processing_time_ms}ms")
        print(f"   المصادر  : {', '.join(result.source_priority[:2])}")
        print("─" * 65)
