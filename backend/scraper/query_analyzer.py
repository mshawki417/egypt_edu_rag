"""
Advanced Arabic Query Analyzer
Production Education RAG Engine v3

Purpose:
- Analyze Arabic educational questions
- Extract curriculum entities
- Build optimized search queries
- Support RAG retrieval pipeline

Compatible with:
- orchestrator.py
- live_scraper.py
- chunker.py
- retriever.py
"""


from __future__ import annotations


import re

from dataclasses import dataclass, field

from collections import Counter

from loguru import logger



# =====================================================
# Education Stages
# =====================================================


STAGE_PATTERNS = {


    "primary": [

        "ابتدائي",

        "ابتدائى",

        "الابتدائي",

        "المرحلة الابتدائية"

    ],


    "preparatory": [

        "اعدادي",

        "إعدادي",

        "الاعدادي",

        "الإعدادي",

        "المرحلة الاعدادية"

    ],


    "secondary": [

        "ثانوي",

        "الثانوية",

        "الثانوية العامة",

        "المرحلة الثانوية"

    ]

}



# =====================================================
# Grade Detection
# =====================================================


GRADE_PATTERNS = {


    "الأول الابتدائي": [

        "اول ابتدائي",

        "الاول ابتدائي",

        "1 ابتدائي"

    ],



    "الثاني الابتدائي": [

        "ثاني ابتدائي",

        "الثاني ابتدائي",

        "2 ابتدائي"

    ],



    "الثالث الابتدائي": [

        "ثالث ابتدائي",

        "الثالث ابتدائي",

        "3 ابتدائي"

    ],



    "الرابع الابتدائي": [

        "رابع ابتدائي",

        "الرابع ابتدائي",

        "4 ابتدائي"

    ],



    "الخامس الابتدائي": [

        "خامس ابتدائي",

        "الخامس ابتدائي",

        "5 ابتدائي"

    ],



    "السادس الابتدائي": [

        "سادس ابتدائي",

        "السادس ابتدائي",

        "6 ابتدائي"

    ],




    "الأول الإعدادي": [

        "اول اعدادي",

        "الاول اعدادي",

        "1 اعدادي"

    ],



    "الثاني الإعدادي": [

        "ثاني اعدادي",

        "الثاني اعدادي",

        "2 اعدادي"

    ],



    "الثالث الإعدادي": [

        "ثالث اعدادي",

        "الثالث اعدادي",

        "3 اعدادي"

    ],



    "الأول الثانوي": [

        "اول ثانوي",

        "الاول ثانوي"

    ],



    "الثاني الثانوي": [

        "ثاني ثانوي",

        "الثاني ثانوي"

    ],



    "الثالث الثانوي": [

        "ثالث ثانوي",

        "الثانوية العامة"

    ]

}




# =====================================================
# Subjects Knowledge Base
# =====================================================


SUBJECTS = {


    "رياضيات": [

        "رياضيات",

        "حساب",

        "جبر",

        "هندسة",

        "تفاضل",

        "تكامل",

        "معادلات",

        "قسمة",

        "ضرب"

    ],



    "فيزياء": [

        "فيزياء",

        "سرعة",

        "قوة",

        "طاقة",

        "حركة",

        "نيوتن"

    ],



    "كيمياء": [

        "كيمياء",

        "ذرة",

        "عنصر",

        "تفاعل",

        "مركب",

        "عضوية"

    ],



    "علوم": [

        "علوم",

        "science",

        "طبيعة"

    ],



    "احياء": [

        "احياء",

        "خلية",

        "وراثة",

        "جسم الانسان"

    ],



    "لغة عربية": [

        "عربي",

        "نحو",

        "بلاغة",

        "شعر",

        "قراءة"

    ],



    "لغة انجليزية": [

        "انجليزي",

        "english",

        "grammar"

    ],



    "تاريخ": [

        "تاريخ",

        "حضارة",

        "ثورة",

        "حرب",

        "اسرات"

    ],



    "جغرافيا": [

        "جغرافيا",

        "مناخ",

        "سكان",

        "خرائط"

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

        "الخلاصة",

        "اختصر"

    ],



    "explanation": [

        "اشرح",

        "شرح",

        "وضح",

        "فسر",

        "كيف"

    ],



    "solution": [

        "حل",

        "مسألة",

        "تمرين",

        "اجابة",

        "إجابة"

    ],



    "definition": [

        "ما هو",

        "من هو",

        "تعريف",

        "عرف"

    ],



    "comparison": [

        "قارن",

        "الفرق",

        "مقارنة"

    ],



    "exam": [

        "امتحان",

        "اختبار",

        "سؤال",

        "درجات"

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
# Metadata Object
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



    entities: dict = field(

        default_factory=dict

    )

# =====================================================
# Text Normalization
# =====================================================


def normalize_text(text: str) -> str:

    """
    Normalize Arabic text for matching and retrieval.
    """

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



    # Remove Arabic tashkeel

    text = re.sub(

        r"[\u064B-\u065F]",

        "",

        text

    )



    # Remove symbols

    text = re.sub(

        r"[^\w\s\u0600-\u06FF]",

        " ",

        text

    )



    # Remove duplicated spaces

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

    """
    Detect first matching entity.
    """

    if not text:

        return None



    for key, values in dictionary.items():


        for value in values:


            if value in text:

                return key



    return None





# =====================================================
# Multi Entity Detection
# =====================================================


def extract_entities(
    text: str,
    dictionary: dict
):

    """
    Extract all matching entities.
    Useful for keywords.
    """

    matches = []



    for key, values in dictionary.items():


        for value in values:


            if value in text:

                matches.append(key)

                break



    return list(
        set(matches)
    )





# =====================================================
# Intent Detection
# =====================================================


def detect_intent(text: str) -> str:


    scores = {}



    for intent, words in INTENT_RULES.items():


        score = 0


        for word in words:


            if word in text:

                score += 1



        scores[intent] = score



    if not scores:

        return "general"



    best_intent = max(

        scores,

        key=scores.get

    )



    if scores[best_intent] == 0:

        return "general"



    return best_intent





# =====================================================
# Stage Detection
# =====================================================


def detect_stage(text: str):


    for stage, words in STAGE_PATTERNS.items():


        for word in words:


            if word in text:

                return stage



    return None






# =====================================================
# Grade Detection
# =====================================================


def detect_grade(text: str):


    for grade, words in GRADE_PATTERNS.items():


        for word in words:


            if word in text:

                return grade



    return None






# =====================================================
# Term Detection
# =====================================================


def detect_term(text: str):


    term_patterns = {


        "الترم الأول": [

            "الترم الاول",

            "الفصل الاول",

            "الفصل الأول"

        ],



        "الترم الثاني": [

            "الترم الثاني",

            "الفصل الثاني",

            "الفصل الثاني"

        ]

    }



    for term, patterns in term_patterns.items():


        for pattern in patterns:


            if pattern in text:

                return term



    return None





# =====================================================
# Year Detection
# =====================================================


def detect_year(text: str):


    result = re.search(

        r"(20\d{2})",

        text

    )



    if result:

        return result.group(1)



    return None






# =====================================================
# Topic Extraction
# =====================================================


def extract_topic(
    text: str,
    subject: str | None = None
):

    """
    Extract educational topic automatically.

    Example:
    شرح الجهاز الهضمي في العلوم

    output:
    الجهاز الهضمي
    """



    stop_words = {


        "شرح",

        "اشرح",

        "درس",

        "منهج",

        "كتاب",

        "ملخص",

        "لخص",

        "المرحله",

        "المرحلة",

        "الثانوي",

        "الاعدادي",

        "الابتدائي",

        "العامة"

    }



    words = text.split()



    topic_words = []



    for word in words:


        if word not in stop_words:

            topic_words.append(word)



    topic = " ".join(

        topic_words[:6]

    )



    return topic.strip() if topic else None





# =====================================================
# Dynamic Keyword Extraction
# =====================================================


def extract_keywords(
    text: str
):

    """
    Extract important search keywords
    without predefined dictionary.
    """



    words = text.split()



    filtered = []



    stop_words = {


        "ما",

        "هو",

        "هي",

        "من",

        "في",

        "على",

        "عن",

        "الى",

        "الي",

        "شرح",

        "اشرح",

        "اريد",

        "اريد"

    }



    for word in words:


        if len(word) < 3:

            continue



        if word in stop_words:

            continue



        filtered.append(word)



    counter = Counter(
        filtered
    )



    return [

        word

        for word, _ in counter.most_common(10)

    ]

# =====================================================
# Source Category Detection
# =====================================================


def detect_source_category(meta: QueryMetadata):


    """
    Determine the best source type.

    Curriculum questions:
        Ministry / EKB sources preferred

    General questions:
        Educational sources + trusted references

    News:
        Latest sources
    """



    if (

        meta.grade

        or meta.stage

        or meta.term

        or meta.intent == "curriculum"

    ):

        return "curriculum"




    if meta.intent == "news":

        return "news"




    if meta.subject:

        return "education"




    return "general"






# =====================================================
# Live Search Decision
# =====================================================


def detect_live_search(text: str):


    realtime_words = [

        "اليوم",

        "الان",

        "حاليا",

        "اخر",

        "جديد",

        "قرار",

        "2026"

    ]



    return any(

        word in text

        for word in realtime_words

    )







# =====================================================
# Confidence Score
# =====================================================


def calculate_confidence(meta: QueryMetadata):


    score = 0



    if meta.subject:

        score += 0.25



    if meta.grade:

        score += 0.20



    if meta.stage:

        score += 0.15



    if meta.topic:

        score += 0.20



    if meta.intent != "general":

        score += 0.10



    if len(meta.keywords) > 2:

        score += 0.10



    return round(

        min(score, 1.0),

        2

    )







# =====================================================
# Search Query Builder
# =====================================================


def build_search_query(meta: QueryMetadata):


    """
    Build optimized query for scraper.

    Example:

    Input:
    شرح الجهاز الهضمي الصف الثاني الاعدادي

    Output:

    شرح الجهاز الهضمي
    علوم
    الثاني الاعدادي
    وزارة التربية والتعليم مصر

    """



    query_parts = []



    # Original question

    query_parts.append(

        meta.raw_question

    )




    if meta.topic:

        query_parts.append(

            meta.topic

        )




    if meta.subject:

        query_parts.append(

            meta.subject

        )




    if meta.grade:

        query_parts.append(

            meta.grade

        )




    if meta.term:

        query_parts.append(

            meta.term

        )




    if meta.year:

        query_parts.append(

            meta.year

        )




    # Curriculum priority

    if meta.source_category == "curriculum":


        query_parts.extend(

            [

                "وزارة التربية والتعليم مصر",

                "كتاب الوزارة",

                "منهج"

            ]

        )




    elif meta.source_category == "education":


        query_parts.append(

            "شرح تعليمي"

        )




    return " ".join(

        dict.fromkeys(

            query_parts

        )

    )








# =====================================================
# Main Analyzer
# =====================================================


def analyze_query(
    question: str
) -> QueryMetadata:


    """
    Main pipeline.

    Converts user question into
    structured retrieval metadata.
    """



    try:


        normalized = normalize_text(

            question

        )



        meta = QueryMetadata(

            raw_question=question,

            normalized=normalized

        )




        # -------------------------
        # Entity Extraction
        # -------------------------


        meta.intent = detect_intent(

            normalized

        )



        meta.subject = detect_from_dictionary(

            normalized,

            SUBJECTS

        )



        meta.stage = detect_stage(

            normalized

        )



        meta.grade = detect_grade(

            normalized

        )



        meta.term = detect_term(

            normalized

        )



        meta.year = detect_year(

            normalized

        )




        # -------------------------
        # Topic Extraction
        # -------------------------


        meta.topic = extract_topic(

            normalized,

            meta.subject

        )




        # -------------------------
        # Dynamic Keywords
        # -------------------------


        meta.keywords = extract_keywords(

            normalized

        )



        # Add detected entities

        for item in [

            meta.subject,

            meta.grade,

            meta.topic

        ]:


            if item:

                meta.keywords.append(

                    item

                )




        meta.keywords = list(

            set(

                meta.keywords

            )

        )





        # -------------------------
        # Source Selection
        # -------------------------


        meta.source_category = detect_source_category(

            meta

        )




        # -------------------------
        # Live Search
        # -------------------------


        meta.needs_live_search = detect_live_search(

            normalized

        )




        # -------------------------
        # Build Query
        # -------------------------


        meta.search_query = build_search_query(

            meta

        )




        # -------------------------
        # Confidence
        # -------------------------


        meta.confidence = calculate_confidence(

            meta

        )



        meta.entities = {


            "subject":

            meta.subject,


            "grade":

            meta.grade,


            "stage":

            meta.stage,


            "topic":

            meta.topic,


            "keywords":

            meta.keywords

        }




        logger.info(

            {

                "query": question,

                "intent": meta.intent,

                "subject": meta.subject,

                "grade": meta.grade,

                "topic": meta.topic,

                "source": meta.source_category,

                "confidence": meta.confidence

            }

        )



        return meta




    except Exception as e:


        logger.exception(

            f"Query analyzer failed: {e}"

        )



        # Safe fallback

        return QueryMetadata(

            raw_question=question,

            normalized=normalize_text(question),

            search_query=question,

            keywords=question.split(),

            confidence=0.1

        )

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


    # New fields

    confidence: float = 0.0


    entities: dict = field(
        default_factory=dict
    )


    keywords: list[str] = field(
        default_factory=list
    )


    search_query: str = ""

    source_category: str = "general"

    needs_live_search: bool = False

def get_source_priority(meta):


    if meta.source_category == "curriculum":

        return [

            "moe.gov.eg",

            "ekb.eg",

            "study.ekb.eg"

        ]



    if meta.subject:

        return [

            "ekb.eg",

            "marefa.org",

            "wikipedia.org"

        ]


    return []

def expand_query(meta):


    expansions = []


    if meta.topic:


        expansions.extend([

            f"شرح {meta.topic}",

            f"خطوات {meta.topic}",

            f"أمثلة {meta.topic}"

        ])



    if meta.subject:


        expansions.append(

            meta.subject

        )



    return expansions

def answer_style(intent):


    styles = {


        "explanation":

        "step_by_step",


        "summary":

        "short_summary",


        "solution":

        "solution_with_steps",


        "definition":

        "direct_answer"


    }


    return styles.get(

        intent,

        "general"

    )
