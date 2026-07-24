"""
Advanced Arabic Query Analyzer
Production Education RAG Engine

Features:
- Curriculum QA
- General Education QA
- Exam Assistant
- Explanation
- Summarization
- Smart Query Expansion
- Automatic Topic Extraction
- Confidence Scoring
"""

from __future__ import annotations


import re

from dataclasses import dataclass, field

from typing import Optional

from loguru import logger



# =====================================================
# Education Stages
# =====================================================


STAGE_PATTERNS = {

    "primary": [

        "ابتدائي",
        "ابتدائيه",
        "الابتدائي",
        "المرحله الابتدائيه",
        "primary"

    ],


    "preparatory": [

        "اعدادي",
        "اعداديه",
        "الاعدادي",
        "المرحله الاعداديه",
        "middle school"

    ],


    "secondary": [

        "ثانوي",
        "ثانويه",
        "الثانويه العامه",
        "المرحله الثانويه",
        "secondary"

    ]

}



# =====================================================
# Grades
# =====================================================


GRADE_PATTERNS = {


    "الأول الابتدائي": [

        "اول ابتدائي",
        "الاول ابتدائي",
        "1 ابتدائي",
        "١ ابتدائي"

    ],


    "الثاني الابتدائي": [

        "ثاني ابتدائي",
        "الثاني ابتدائي",
        "2 ابتدائي",
        "٢ ابتدائي"

    ],


    "الثالث الابتدائي": [

        "ثالث ابتدائي",
        "الثالث ابتدائي",
        "3 ابتدائي",
        "٣ ابتدائي"

    ],


    "الرابع الابتدائي": [

        "رابع ابتدائي",
        "الرابع ابتدائي",
        "4 ابتدائي",
        "٤ ابتدائي"

    ],


    "الخامس الابتدائي": [

        "خامس ابتدائي",
        "الخامس ابتدائي",
        "5 ابتدائي",
        "٥ ابتدائي"

    ],


    "السادس الابتدائي": [

        "سادس ابتدائي",
        "السادس ابتدائي",
        "6 ابتدائي",
        "٦ ابتدائي"

    ],



    "الأول الإعدادي": [

        "اول اعدادي",
        "الاول اعدادي",
        "1 اعدادي",
        "١ اعدادي"

    ],


    "الثاني الإعدادي": [

        "ثاني اعدادي",
        "الثاني اعدادي",
        "2 اعدادي",
        "٢ اعدادي"

    ],


    "الثالث الإعدادي": [

        "ثالث اعدادي",
        "الثالث اعدادي",
        "3 اعدادي",
        "٣ اعدادي"

    ],



    "الأول الثانوي": [

        "اول ثانوي",
        "الاول ثانوي",
        "1 ثانوي"

    ],


    "الثاني الثانوي": [

        "ثاني ثانوي",
        "الثاني ثانوي",
        "2 ثانوي"

    ],


    "الثالث الثانوي": [

        "ثالث ثانوي",
        "الثالث ثانوي",
        "الثانويه العامه",
        "الثانويه العامة"

    ]

}



# =====================================================
# Subjects Intelligence Dictionary
# =====================================================


SUBJECTS = {


    "رياضيات": [

        "رياضيات",
        "حساب",
        "جبر",
        "هندسه",
        "هندسة",
        "تفاضل",
        "تكامل",
        "معادلات",
        "قسمة",
        "ضرب",
        "كسور",
        "احتمالات"

    ],



    "فيزياء": [

        "فيزياء",
        "حركه",
        "حركة",
        "سرعه",
        "سرعة",
        "قوه",
        "قوة",
        "طاقه",
        "طاقة",
        "تسارع",
        "نيوتن",
        "قانون"

    ],



    "كيمياء": [

        "كيمياء",
        "ذره",
        "ذرة",
        "تفاعل",
        "عنصر",
        "مركب",
        "احماض",
        "عضويه",
        "عضوية"

    ],



    "علوم": [

        "علوم",
        "science",
        "biology",
        "طبيعه",
        "طبيعة"

    ],



    "احياء": [

        "احياء",
        "خلية",
        "جسم الانسان",
        "تشريح",
        "وراثه",
        "وراثة"

    ],



    "لغة عربية": [

        "عربي",
        "لغة عربية",
        "نحو",
        "بلاغه",
        "بلاغة",
        "شعر",
        "ادب"

    ],



    "لغة انجليزية": [

        "انجليزي",
        "english",
        "grammar"

    ],



    "تاريخ": [

        "تاريخ",
        "حضاره",
        "حضارة",
        "ثورة",
        "حرب",
        "اسرات",
        "اسر"

    ],



    "جغرافيا": [

        "جغرافيا",
        "مناخ",
        "سكان",
        "بيئه",
        "بيئة"

    ]

}



# =====================================================
# Topic Expansion Map
# =====================================================


TOPIC_MAP = {


    "السرعة": [

        "السرعة",
        "velocity",
        "speed",
        "حركة",
        "مسافة",
        "زمن"

    ],


    "القسمة المطولة": [

        "قسمة",
        "قسمة طويلة",
        "القسمة المطولة",
        "رياضيات"

    ],


    "قوانين نيوتن": [

        "نيوتن",
        "القانون الاول",
        "القانون الثاني",
        "القوة",
        "التسارع"

    ],


    "الجهاز الهضمي": [

        "جهاز هضمي",
        "معدة",
        "امعاء",
        "هضم",
        "علوم"

    ],


    "الكيمياء العضوية": [

        "عضوية",
        "هيدروكربونات",
        "كربون",
        "كيمياء"

    ]

}



# =====================================================
# Intent Rules
# =====================================================


INTENT_RULES = {


    "summary": [

        "لخص",
        "ملخص",
        "تلخيص",
        "الخلاصة"

    ],


    "explanation": [

        "اشرح",
        "شرح",
        "وضح",
        "فسر",
        "كيف يعمل",
        "ما السبب"

    ],


    "solution": [

        "حل",
        "مسألة",
        "تمرين",
        "احسب",
        "اوجد"

    ],


    "definition": [

        "ما هو",
        "عرف",
        "تعريف",
        "من هو"

    ],


    "comparison": [

        "قارن",
        "الفرق",
        "مقارنة"

    ],


    "exam": [

        "امتحان",
        "اختبار",
        "درجات",
        "نماذج"

    ],


    "news": [

        "قرار",
        "خبر",
        "اليوم",
        "اخر",
        "جديد"

    ],


    "curriculum": [

        "منهج",
        "كتاب",
        "درس",
        "باب",
        "وحدة"

    ]

}

# =====================================================
# Query Metadata
# =====================================================


@dataclass
class QueryMetadata:

    raw_question: str

    normalized: str = ""

    intent: str = "general"

    subject: str | None = None

    stage: str | None = None

    grade: str | None = None

    term: str | None = None

    year: str | None = None

    topic: str | None = None

    keywords: list[str] = field(
        default_factory=list
    )

    search_query: str = ""

    source_category: str = "general"

    needs_live_search: bool = False

    confidence: float = 0.0

    detected_entities: dict = field(
        default_factory=dict
    )



# =====================================================
# Arabic Normalization
# =====================================================


def normalize_text(text: str) -> str:

    if not text:
        return ""


    text = text.lower()


    replacements = {

        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي"

    }


    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )


    # remove tashkeel

    text = re.sub(

        r"[\u064B-\u065F]",

        "",

        text

    )


    # remove symbols

    text = re.sub(

        r"[^\w\s\u0600-\u06ff]",

        " ",

        text

    )


    # normalize spaces

    text = re.sub(

        r"\s+",

        " ",

        text

    )


    return text.strip()



# =====================================================
# Dictionary Detection
# =====================================================


def detect_from_dictionary(
    text: str,
    dictionary: dict
):

    matches = []


    for key, values in dictionary.items():

        for value in values:

            if value in text:

                matches.append(key)

                break



    return matches[0] if matches else None




# =====================================================
# Intent Detection With Score
# =====================================================


def detect_intent(text: str):

    scores = {}


    for intent, words in INTENT_RULES.items():

        score = 0


        for word in words:

            if word in text:

                score += 1


        scores[intent] = score



    best_intent = max(
        scores,
        key=scores.get
    )



    if scores[best_intent] == 0:

        return "general"



    return best_intent



# =====================================================
# Detect Year
# =====================================================


def detect_year(text: str):

    match = re.search(

        r"(20\d{2})",

        text

    )


    if match:

        return match.group(1)


    return None



# =====================================================
# Detect Term
# =====================================================


def detect_term(text: str):


    if any(
        x in text
        for x in [
            "الترم الاول",
            "الفصل الاول",
            "الفصل الدراسي الاول"
        ]
    ):

        return "الترم الأول"



    if any(
        x in text
        for x in [
            "الترم الثاني",
            "الفصل الثاني",
            "الفصل الدراسي الثاني"
        ]
    ):

        return "الترم الثاني"



    return None



# =====================================================
# Dynamic Keyword Extraction
# =====================================================


def extract_keywords(text: str):

    words = text.split()


    stop_words = {

        "ما",
        "هو",
        "هي",
        "في",
        "من",
        "عن",
        "الى",
        "على",
        "شرح",
        "اشرح",
        "اريد",
        "عاوز",
        "كيف",
        "هل"

    }


    filtered = [

        w

        for w in words

        if w not in stop_words

        and len(w) > 2

    ]


    frequency = Counter(filtered)


    return [

        word

        for word, count

        in frequency.most_common(10)

    ]

# =====================================================
# Topic Extraction
# =====================================================


def extract_topic(
    text: str,
    subject: str | None
):

    keywords = extract_keywords(text)


    if subject:

        keywords = [

            k

            for k in keywords

            if k != subject

        ]

    if keywords:

        return " ".join(
            keywords[:3]
        )


    return None

# =====================================================
# Source Category Detection
# =====================================================


def detect_source_category(meta: QueryMetadata):


    # Curriculum questions
    if (
        meta.grade
        or meta.term
        or meta.subject
        or meta.intent == "curriculum"
    ):

        return "curriculum"



    # Exam related

    if meta.intent == "exam":

        return "exam"



    # Current information

    if meta.needs_live_search:

        return "news"



    return "general"





# =====================================================
# Live Search Decision
# =====================================================


def should_use_live_search(text: str):


    triggers = [

        "اليوم",

        "الان",

        "اخر",

        "جديد",

        "قرار",

        "2025",

        "2026",

        "وزارة",

        "نتيجة"

    ]



    return any(

        word in text

        for word in triggers

    )





# =====================================================
# Build Smart Search Query
# =====================================================


def build_search_query(meta: QueryMetadata):


    query_parts = []



    # Original question

    query_parts.append(

        meta.raw_question

    )



    # Subject

    if meta.subject:

        query_parts.append(

            meta.subject

        )



    # Grade

    if meta.grade:

        query_parts.append(

            meta.grade

        )



    # Stage

    if meta.stage:

        query_parts.append(

            "المرحلة " + meta.stage

        )



    # Term

    if meta.term:

        query_parts.append(

            meta.term

        )



    # Topic extracted automatically

    if meta.topic:

        query_parts.append(

            meta.topic

        )



    # Curriculum priority

    if meta.source_category == "curriculum":


        query_parts.extend(

            [

                "وزارة التربية والتعليم مصر",

                "كتاب الوزارة",

                "المنهج المصري"

            ]

        )



    # Exam priority

    if meta.source_category == "exam":

        query_parts.extend(

            [

                "امتحانات وزارة التربية والتعليم",

                "نماذج امتحانات"

            ]

        )



    return " ".join(

        query_parts

    )





# =====================================================
# Confidence Calculation
# =====================================================


def calculate_confidence(meta: QueryMetadata):


    score = 0



    if meta.subject:

        score += 0.25



    if meta.grade:

        score += 0.25



    if meta.topic:

        score += 0.20



    if meta.intent != "general":

        score += 0.15



    if meta.term:

        score += 0.15



    return round(

        min(score, 1.0),

        2

    )





# =====================================================
# Main Analyzer
# =====================================================


def analyze_query(question: str):


    normalized = normalize_text(

        question

    )



    meta = QueryMetadata(

        raw_question=question,

        normalized=normalized

    )



    # Intent

    meta.intent = detect_intent(

        normalized

    )



    # Subject

    meta.subject = detect_from_dictionary(

        normalized,

        SUBJECTS

    )



    # Grade

    meta.grade = detect_from_dictionary(

        normalized,

        GRADE_PATTERNS

    )



    # Stage

    for stage, patterns in STAGE_PATTERNS.items():


        if any(

            p in normalized

            for p in patterns

        ):

            meta.stage = stage

            break



    # Term

    meta.term = detect_term(

        normalized

    )



    # Year

    meta.year = detect_year(

        normalized

    )



    # Keywords

    meta.keywords = extract_keywords(

        normalized

    )



    # Topic

    meta.topic = extract_topic(

        normalized,

        meta.subject

    )



    # Live search

    meta.needs_live_search = should_use_live_search(

        normalized

    )



    # Source category

    meta.source_category = detect_source_category(

        meta

    )



    # Confidence

    meta.confidence = calculate_confidence(

        meta

    )



    # Entities for debugging

    meta.detected_entities = {


        "subject":
            meta.subject,


        "grade":
            meta.grade,


        "stage":
            meta.stage,


        "term":
            meta.term,


        "topic":
            meta.topic,


        "year":
            meta.year

    }



    # Search Query

    meta.search_query = build_search_query(

        meta

    )



    logger.info(

        {

            "query": question,

            "intent": meta.intent,

            "subject": meta.subject,

            "grade": meta.grade,

            "topic": meta.topic,

            "source": meta.source_category,

            "confidence": meta.confidence,

            "search_query": meta.search_query

        }

    )

    return meta
