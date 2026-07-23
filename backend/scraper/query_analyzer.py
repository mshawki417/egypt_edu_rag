"""
Advanced Arabic Query Analyzer
Production Education RAG Engine

Designed for:
- Curriculum QA
- General Education QA
- Exam Assistant
- Summarization
- Explanation
"""


from __future__ import annotations


import re

from dataclasses import dataclass, field

from loguru import logger





# =====================================================
# Education Levels
# =====================================================


STAGE_PATTERNS={


    "primary":
    [
        "ابتدائي",
        "ابتدائى",
        "الابتدائي"
    ],


    "preparatory":
    [
        "اعدادي",
        "إعدادي",
        "الاعدادي"
    ],


    "secondary":
    [
        "ثانوي",
        "الثانوية العامة"
    ]

}





# =====================================================
# Grade Detection
# =====================================================


GRADE_PATTERNS={


"الأول الابتدائي":
[
"اول ابتدائي",
"الاول ابتدائي",
"1 ابتدائي"
],


"الثاني الابتدائي":
[
"ثاني ابتدائي",
"الثاني ابتدائي",
"2 ابتدائي"
],


"الثالث الابتدائي":
[
"ثالث ابتدائي",
"الثالث ابتدائي",
"3 ابتدائي"
],


"الرابع الابتدائي":
[
"رابع ابتدائي",
"الرابع ابتدائي",
"4 ابتدائي"
],


"الخامس الابتدائي":
[
"خامس ابتدائي",
"الخامس ابتدائي",
"5 ابتدائي"
],


"السادس الابتدائي":
[
"سادس ابتدائي",
"السادس ابتدائي",
"6 ابتدائي"
],



"الأول الإعدادي":
[
"اول اعدادي",
"الاول اعدادي",
"1 اعدادي"
],


"الثاني الإعدادي":
[
"ثاني اعدادي",
"الثاني اعدادي",
"2 اعدادي"
],


"الثالث الإعدادي":
[
"ثالث اعدادي",
"الثالث اعدادي",
"3 اعدادي"
],



"الأول الثانوي":
[
"اول ثانوي",
"الاول ثانوي"
],


"الثاني الثانوي":
[
"ثاني ثانوي",
"الثاني ثانوي"
],


"الثالث الثانوي":
[
"ثالث ثانوي",
"الثانوية العامة"
]

}





# =====================================================
# Subjects
# =====================================================


SUBJECTS={


"رياضيات":
[
"رياضيات",
"حساب",
"جبر",
"هندسة",
"تفاضل",
"تكامل",
"قسمة",
"ضرب"
],



"فيزياء":
[
"فيزياء",
"سرعة",
"قوة",
"طاقة",
"حركة"
],



"كيمياء":
[
"كيمياء",
"تفاعل",
"ذرة"
],



"علوم":
[
"علوم",
"biology",
"science"
],



"احياء":
[
"احياء",
"خلية",
"جسم الانسان"
],



"لغة عربية":
[
"عربي",
"نحو",
"بلاغة",
"شعر"
],



"لغة انجليزية":
[
"انجليزي",
"english"
],



"تاريخ":
[
"تاريخ",
"ثورة",
"حرب"
],



"جغرافيا":
[
"جغرافيا",
"مناخ",
"سكان"
]

}





# =====================================================
# Intent
# =====================================================


INTENT_RULES={


"summary":
[
"لخص",
"ملخص",
"تلخيص",
"الخلاصة"
],



"explanation":
[
"اشرح",
"شرح",
"وضح",
"فسر",
"كيف"
],



"solution":
[
"حل",
"مسألة",
"تمرين",
"مثال"
],



"definition":
[
"من هو",
"من صاحب",
"ما هو",
"عرف",
"تعريف"
],



"comparison":
[
"قارن",
"الفرق",
"مقارنة"
],



"exam":
[
"امتحان",
"اختبار",
"نتيجة",
"درجات"
],



"news":
[
"قرار",
"خبر",
"اليوم",
"اخر",
"جديد"
],



"curriculum":
[
"منهج",
"كتاب",
"درس",
"باب",
"وحدة"
]

}





# =====================================================
# Metadata
# =====================================================


@dataclass
class QueryMetadata:


    raw_question:str


    normalized:str=""


    intent:str="general"


    subject:str|None=None


    stage:str|None=None


    grade:str|None=None


    term:str|None=None


    year:str|None=None


    topic:str|None=None


    search_query:str=""


    keywords:list[str]=field(
        default_factory=list
    )


    source_category:str="general"


    needs_live_search:bool=False





# =====================================================
# Normalize
# =====================================================


def normalize_text(text):


    text=text.lower()


    replacements={

        "أ":"ا",
        "إ":"ا",
        "آ":"ا",
        "ى":"ي",
        "ة":"ه"

    }


    for a,b in replacements.items():

        text=text.replace(a,b)



    text=re.sub(

        r"[^\w\s\u0600-\u06ff]",

        " ",

        text

    )


    text=re.sub(

        r"\s+",

        " ",

        text

    )


    return text.strip()





# =====================================================
# Detect
# =====================================================


def detect_from_dictionary(text,dictionary):


    for key,values in dictionary.items():


        for value in values:


            if value in text:

                return key


    return None






def detect_intent(text):


    scores={}



    for intent,words in INTENT_RULES.items():

        scores[intent]=sum(

            1

            for w in words

            if w in text

        )



    best=max(

        scores,

        key=scores.get

    )



    if scores[best]==0:

        return "general"



    return best






def detect_year(text):


    result=re.search(

        r"(20\d{2})",

        text

    )


    return result.group(1) if result else None





def detect_term(text):


    if "الترم الاول" in text:

        return "الترم الأول"


    if "الترم الثاني" in text:

        return "الترم الثاني"


    return None





# =====================================================
# Main
# =====================================================


def analyze_query(question:str):


    text=normalize_text(question)



    meta=QueryMetadata(

        raw_question=question,

        normalized=text

    )



    meta.intent=detect_intent(text)



    meta.subject=detect_from_dictionary(

        text,

        SUBJECTS

    )




    meta.grade=detect_from_dictionary(

        text,

        GRADE_PATTERNS

    )




    meta.term=detect_term(text)



    meta.year=detect_year(text)



    for stage,words in STAGE_PATTERNS.items():

        if any(

            w in text

            for w in words

        ):

            meta.stage=stage

            break





    # -------------------------------
    # Source Category
    # -------------------------------


    if meta.grade or meta.term:

        meta.source_category="curriculum"


    elif meta.intent=="news":

        meta.source_category="news"


    else:

        meta.source_category="general"






    # -------------------------------
    # Live Search
    # -------------------------------


    meta.needs_live_search=any(

        x in text

        for x in [

            "اليوم",

            "الان",

            "اخر",

            "قرار",

            "2026"

        ]

    )






    # -------------------------------
    # Smart Query Builder
    # -------------------------------


    query=[

        question

    ]



    if meta.subject:

        query.append(

            meta.subject

        )



    if meta.grade:

        query.append(

            meta.grade

        )



    if meta.term:

        query.append(

            meta.term

        )




    if meta.source_category=="curriculum":


        query.append(

            "منهج وزارة التربية والتعليم مصر"

        )



    meta.search_query=" ".join(

        query

    )




    meta.keywords=[

        x

        for x in [

            meta.subject,

            meta.grade,

            meta.term,

            meta.intent

        ]

        if x

    ]




    logger.info(

        {

            "query":question,

            "intent":meta.intent,

            "subject":meta.subject,

            "grade":meta.grade,

            "source":meta.source_category

        }

    )



    return meta
