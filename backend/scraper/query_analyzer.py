"""
Advanced Arabic Query Analyzer
Production version for Real-Time RAG
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from loguru import logger



# =====================================
# Grade Detection
# =====================================


GRADE_PATTERNS = {


    # Primary
    r"(اول|الأول|1|١)\s*(ابتدائي|ابتدائى|صف اول|الصف الاول)":
        "الأول الابتدائي",

    r"(ثاني|الثاني|2|٢)\s*(ابتدائي|ابتدائى|صف ثاني|الصف الثاني)":
        "الثاني الابتدائي",

    r"(ثالث|الثالث|3|٣)\s*(ابتدائي|ابتدائى|صف ثالث|الصف الثالث)":
        "الثالث الابتدائي",

    r"(رابع|الرابع|4|٤)\s*(ابتدائي|ابتدائى|صف رابع|الصف الرابع)":
        "الرابع الابتدائي",

    r"(خامس|الخامس|5|٥)\s*(ابتدائي|ابتدائى|صف خامس|الصف الخامس)":
        "الخامس الابتدائي",

    r"(سادس|السادس|6|٦)\s*(ابتدائي|ابتدائى|صف سادس|الصف السادس)":
        "السادس الابتدائي",



    # Preparatory

    r"(اول|الأول|1|١)\s*(اعدادي|إعدادي|الصف الاول)":
        "الأول الإعدادي",

    r"(ثاني|الثاني|2|٢)\s*(اعدادي|إعدادي|الصف الثاني)":
        "الثاني الإعدادي",

    r"(ثالث|الثالث|3|٣)\s*(اعدادي|إعدادي|الصف الثالث)":
        "الثالث الإعدادي",



    # Secondary

    r"(اول|الأول|1|١)\s*(ثانوي)":
        "الأول الثانوي",

    r"(ثاني|الثاني|2|٢)\s*(ثانوي)":
        "الثاني الثانوي",

    r"(ثالث|الثالث|3|٣)\s*(ثانوي|الثانوية العامة)":
        "الثالث الثانوي"

}





# =====================================
# Subject Detection
# =====================================


SUBJECT_PATTERNS = {


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





# =====================================
# Intent Detection
# =====================================


INTENTS = {


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
        "خبر",
        "اخر"
    ],



    "curriculum":
    [
        "منهج",
        "كتاب",
        "شرح",
        "درس",
        "وحدة",
        "باب",
        "مادة",
        "محتوى",
        "صف"
    ]

}





# =====================================
# Metadata
# =====================================


@dataclass
class QueryMetadata:


    raw_question: str

    intent: str = "curriculum"

    grade: str | None = None

    subject: str | None = None

    search_query: str = ""

    keywords: list[str] = field(
        default_factory=list
    )

    needs_live_search: bool = False

    source_category: str = "curriculum"







# =====================================
# Text Normalize
# =====================================


def normalize_text(text:str):


    text = text.lower()



    replacements = {

        "أ":"ا",
        "إ":"ا",
        "آ":"ا",
        "ة":"ه",
        "ى":"ي"

    }



    for old,new in replacements.items():

        text=text.replace(
            old,
            new
        )



    text=re.sub(
        r"[ـ]",
        "",
        text
    )


    text=re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()







# =====================================
# Intent
# =====================================


def detect_intent(text):


    scores={}


    for intent,words in INTENTS.items():


        scores[intent]=sum(

            1

            for word in words

            if word in text

        )



    best=max(

        scores,

        key=scores.get

    )


    # default education query

    if scores[best]==0:

        return "curriculum"



    return best







# =====================================
# Main Analyzer
# =====================================


def analyze_query(question:str):


    text=normalize_text(question)



    meta=QueryMetadata(

        raw_question=question

    )



    # Grade

    for pattern,value in GRADE_PATTERNS.items():


        if re.search(

            pattern,

            text

        ):


            meta.grade=value

            break





    # Subject

    for subject,words in SUBJECT_PATTERNS.items():


        if any(

            word in text

            for word in words

        ):


            meta.subject=subject

            break





    # Intent

    meta.intent=detect_intent(

        text

    )






    # Live Search

    meta.needs_live_search=any(

        word in text

        for word in [

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





    # Build Search Query

    query_parts=[

        question,

        meta.subject,

        meta.grade,

        "وزارة التربية والتعليم",

        "منهج مصر"

    ]



    meta.search_query=" ".join(

        x

        for x in query_parts

        if x

    )





    meta.keywords=[

        x

        for x in [

            meta.subject,

            meta.grade,

            meta.intent

        ]

        if x

    ]





    logger.info(

        f"Intent={meta.intent} | Subject={meta.subject} | Grade={meta.grade}"

    )



    return meta
