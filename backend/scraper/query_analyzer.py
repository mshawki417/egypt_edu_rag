"""
Advanced Arabic Query Analyzer
Production version for Real-Time RAG
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from loguru import logger



GRADE_PATTERNS = {

    r"(اول|الأول|1)\s*(ابتدائي|ابتدائى)":
        "الأول الابتدائي",

    r"(ثاني|الثاني|2)\s*(ابتدائي|ابتدائى)":
        "الثاني الابتدائي",

    r"(ثالث|الثالث|3)\s*(ابتدائي|ابتدائى)":
        "الثالث الابتدائي",

    r"(رابع|الرابع|4)\s*(ابتدائي|ابتدائى)":
        "الرابع الابتدائي",

    r"(خامس|الخامس|5)\s*(ابتدائي|ابتدائى)":
        "الخامس الابتدائي",

    r"(سادس|السادس|6)\s*(ابتدائي|ابتدائى)":
        "السادس الابتدائي",


    r"(اول|الأول|1)\s*(اعدادي|إعدادي)":
        "الأول الإعدادي",

    r"(ثاني|الثاني|2)\s*(اعدادي|إعدادي)":
        "الثاني الإعدادي",

    r"(ثالث|الثالث|3)\s*(اعدادي|إعدادي)":
        "الثالث الإعدادي",


    r"(اول|الأول|1)\s*(ثانوي)":
        "الأول الثانوي",

    r"(ثاني|الثاني|2)\s*(ثانوي)":
        "الثاني الثانوي",

    r"(ثالث|الثالث|3)\s*(ثانوي)|الثانوية العامة":
        "الثالث الثانوي"

}



SUBJECT_PATTERNS={


"رياضيات":
[
"رياضيات",
"حساب",
"جبر",
"هندسة",
"تفاضل",
"تكامل"
],


"علوم":
[
"علوم",
"science"
],


"فيزياء":
[
"فيزياء",
"physics"
],


"كيمياء":
[
"كيمياء",
"chemistry"
],


"أحياء":
[
"احياء",
"أحياء",
"biology"
],


"لغة عربية":
[
"عربي",
"لغة عربية",
"نحو",
"بلاغة"
],


"لغة إنجليزية":
[
"انجليزي",
"english"
],


"تاريخ":
[
"تاريخ",
"history"
],


"دراسات اجتماعية":
[
"دراسات",
"جغرافيا"
]

}




INTENTS={


"exam":
[
"امتحان",
"اختبار",
"جدول",
"نتيجة",
"درجات",
"موعد"
],


"news":
[
"قرار",
"تحديث",
"جديد",
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

    search_query:str=""

    keywords:list[str]=field(default_factory=list)

    needs_live_search:bool=False

    source_category:str="curriculum"





def normalize_text(text):


    text=text.lower()


    text=re.sub(
        r"[أإآ]",
        "ا",
        text
    )


    text=text.replace(
        "ة",
        "ه"
    )


    text=text.replace(
        "ى",
        "ي"
    )


    text=re.sub(
        r"ـ",
        "",
        text
    )


    text=re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()





def detect_intent(text):


    scores={}


    for intent,words in INTENTS.items():

        score=sum(
            1 for w in words
            if w in text
        )

        scores[intent]=score



    return max(
        scores,
        key=scores.get
    )





def analyze_query(question):


    text=normalize_text(question)


    meta=QueryMetadata(
        raw_question=question
    )



    for pattern,value in GRADE_PATTERNS.items():

        if re.search(pattern,text):

            meta.grade=value
            break



    for subject,words in SUBJECT_PATTERNS.items():

        if any(
            w in text
            for w in words
        ):

            meta.subject=subject
            break



    meta.intent=detect_intent(text)



    meta.needs_live_search=any(

        x in text

        for x in [

            "اليوم",
            "الان",
            "اخر",
            "قرار",
            "2025",
            "2026"

        ]

    )



    if meta.intent=="exam":

        meta.source_category="exams"

    elif meta.intent=="news":

        meta.source_category="news"



    query_parts=[

        question,

        meta.subject,

        meta.grade,

        meta.term if hasattr(meta,"term") else None,

        "منهج مصر"

    ]



    meta.search_query=" ".join(

        x for x in query_parts

        if x

    )


    meta.keywords=[

        x for x in [

            meta.subject,

            meta.grade

        ]

        if x

    ]


    logger.info(
        f"Intent={meta.intent} | Subject={meta.subject} | Grade={meta.grade}"
    )


    return meta
