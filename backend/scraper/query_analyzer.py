"""
Advanced Arabic Query Analyzer
Optimized for Real-Time RAG
"""


from __future__ import annotations


import re

from dataclasses import dataclass, field

from loguru import logger



# ============================
# Patterns
# ============================


GRADE_PATTERNS = {

    r"(اول|الأول|1)\s*(ابتدائي|ابتدائي)": 
        "الأول الابتدائي",

    r"(ثاني|الثاني|2)\s*(ابتدائي)":
        "الثاني الابتدائي",


    r"(ثالث|الثالث|3)\s*(اعدادي|الإعدادي)":
        "الثالث الإعدادي",


    r"(اول|الأول|1)\s*(ثانوي|الثانوي)":
        "الأول الثانوي",


    r"(ثاني|الثاني|2)\s*(ثانوي|الثانوي)":
        "الثاني الثانوي",


    r"(ثالث|الثالث|3)\s*(ثانوي|الثانوي)|الثانوية العامة":
        "الثالث الثانوي",

}





SUBJECT_PATTERNS = {


    r"رياضيات|حساب|جبر|هندسة|math":
        "رياضيات",


    r"علوم|science":
        "علوم",


    r"فيزياء|physics":
        "فيزياء",


    r"كيمياء|chemistry":
        "كيمياء",


    r"أحياء|احياء|biology":
        "أحياء",


    r"عربي|لغة عربية|نحو":
        "لغة عربية",


    r"انجليزي|إنجليزي|english":
        "لغة إنجليزية",


    r"تاريخ|history":
        "تاريخ",

}





TERM_PATTERNS={


    r"ترم\s*اول|الفصل\s*الأول":
        "الأول",


    r"ترم\s*ثاني|الفصل\s*الثاني":
        "الثاني"

}




# ============================
# Intent Detection
# ============================


INTENTS={


    "exam":

    [
        "امتحان",
        "اختبار",
        "موعد",
        "جدول",
        "درجات"
    ],


    "news":

    [
        "قرار",
        "جديد",
        "تحديث",
        "وزارة",
        "خبر"
    ],


    "curriculum":

    [
        "منهج",
        "كتاب",
        "شرح",
        "درس",
        "وحدة"
    ]

}




@dataclass
class QueryMetadata:


    raw_question:str


    intent:str="curriculum"


    grade:str|None=None


    subject:str|None=None


    term:str|None=None


    source_category:str="curriculum"


    keywords:list[str]=field(
        default_factory=list
    )


    search_query:str=""


    needs_live_search:bool=False





# ============================
# Normalize
# ============================


def normalize_text(text):


    text=text.lower()


    replacements={

        "أ":"ا",

        "إ":"ا",

        "آ":"ا"

    }


    for a,b in replacements.items():

        text=text.replace(a,b)


    text=re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()





# ============================
# Intent
# ============================


def detect_intent(text):


    for intent,words in INTENTS.items():


        for w in words:

            if w in text:

                return intent


    return "curriculum"





# ============================
# Analyzer
# ============================


def analyze_query(
    question:str
):


    original=question


    text=normalize_text(
        question
    )


    meta=QueryMetadata(
        raw_question=original
    )



    # Grade

    for pattern,value in GRADE_PATTERNS.items():

        if re.search(pattern,text):

            meta.grade=value

            break



    # Subject

    for pattern,value in SUBJECT_PATTERNS.items():

        if re.search(pattern,text):

            meta.subject=value

            break



    # Term

    for pattern,value in TERM_PATTERNS.items():

        if re.search(pattern,text):

            meta.term=value

            break



    # Intent

    meta.intent=detect_intent(
        text
    )



    # Live Search

    live_words=[

        "اليوم",

        "الان",

        "جديد",

        "اخر",

        "قرار",

        "2026",

        "2025"

    ]


    meta.needs_live_search = any(

        w in text

        for w in live_words

    )



    # Category


    if meta.intent=="news":

        meta.source_category="news"


    elif meta.intent=="exam":

        meta.source_category="exams"


    elif meta.subject:

        meta.source_category="curriculum"


    else:

        meta.source_category="curriculum"




    # Search Query

    query_parts=[


        meta.subject,


        meta.grade,


        meta.term,


        meta.intent,


        "التعليم المصري"


    ]


    meta.search_query=" ".join(

        x for x in query_parts

        if x

    )



    meta.keywords=[

        x for x in [

            meta.subject,

            meta.grade,

            meta.term

        ]

        if x

    ]



    logger.info(

        f"""
        Query:
        {question}

        Intent:
        {meta.intent}

        Grade:
        {meta.grade}

        Subject:
        {meta.subject}

        Live:
        {meta.needs_live_search}
        """

    )



    return meta
