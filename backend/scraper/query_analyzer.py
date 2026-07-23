"""
Advanced Arabic Query Analyzer
Production Education RAG Version
"""

from __future__ import annotations


import re

from dataclasses import dataclass, field

from loguru import logger



# ======================================================
# Grade / Stage Detection
# ======================================================


GRADE_PATTERNS = {


    # Primary

    r"(الصف\s*)?(الاول|اول|1|١)\s*(ابتدائي|ابتدائى)":
        "الأول الابتدائي",


    r"(الصف\s*)?(الثاني|ثاني|2|٢)\s*(ابتدائي|ابتدائى)":
        "الثاني الابتدائي",


    r"(الصف\s*)?(الثالث|ثالث|3|٣)\s*(ابتدائي|ابتدائى)":
        "الثالث الابتدائي",


    r"(الصف\s*)?(الرابع|رابع|4|٤)\s*(ابتدائي|ابتدائى)":
        "الرابع الابتدائي",


    r"(الصف\s*)?(الخامس|خامس|5|٥)\s*(ابتدائي|ابتدائى)":
        "الخامس الابتدائي",


    r"(الصف\s*)?(السادس|سادس|6|٦)\s*(ابتدائي|ابتدائى)":
        "السادس الابتدائي",



    # Preparatory

    r"(الصف\s*)?(الاول|اول|1|١)\s*(اعدادي|إعدادي)":
        "الأول الإعدادي",


    r"(الصف\s*)?(الثاني|ثاني|2|٢)\s*(اعدادي|إعدادي)":
        "الثاني الإعدادي",


    r"(الصف\s*)?(الثالث|ثالث|3|٣)\s*(اعدادي|إعدادي)":
        "الثالث الإعدادي",



    # Secondary

    r"(الصف\s*)?(الاول|اول|1|١)\s*(ثانوي)":
        "الأول الثانوي",


    r"(الصف\s*)?(الثاني|ثاني|2|٢)\s*(ثانوي)":
        "الثاني الثانوي",


    r"(الصف\s*)?(الثالث|ثالث|3|٣)\s*(ثانوي|ثانويه عامه)":
        "الثالث الثانوي"


}




# ======================================================
# Semester Detection
# ======================================================


TERM_PATTERNS={


    r"(الترم الاول|الفصل الاول|ترم اول)":
        "الترم الأول",


    r"(الترم الثاني|الفصل الثاني|ترم ثاني)":
        "الترم الثاني"

}




# ======================================================
# Subjects
# ======================================================


SUBJECT_PATTERNS={


"رياضيات":
[
"رياضيات",
"حساب",
"جبر",
"هندسه",
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



"احياء":
[
"احياء",
"biology"
],



"لغة عربية":
[
"عربي",
"لغة عربية",
"نحو",
"بلاغه"
],



"لغة انجليزية":
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




# ======================================================
# Intent
# ======================================================


INTENTS={



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
"فسر"
],



"exercise":
[
"حل",
"تمارين",
"مسائل",
"امثلة"
],



"exam":
[
"امتحان",
"اختبار",
"نتيجة",
"درجات",
"جدول"
],



"definition":
[
"من هو",
"من صاحب",
"ما هو",
"عرف"
],



"news":
[
"قرار",
"اخر",
"جديد",
"وزارة",
"خبر"
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




# ======================================================
# Metadata
# ======================================================


@dataclass
class QueryMetadata:


    raw_question:str


    intent:str="curriculum"


    grade:str|None=None


    subject:str|None=None


    term:str|None=None


    search_query:str=""


    keywords:list[str]=field(
        default_factory=list
    )


    needs_live_search:bool=False


    source_category:str="education"





# ======================================================
# Normalize
# ======================================================


def normalize_text(text):


    text=text.lower()



    replacements={

        "أ":"ا",
        "إ":"ا",
        "آ":"ا",
        "ى":"ي"

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





# ======================================================
# Detect Intent
# ======================================================


def detect_intent(text):


    scores={}



    for name,words in INTENTS.items():


        scores[name]=sum(

            1

            for w in words

            if w in text

        )



    result=max(

        scores,

        key=scores.get

    )



    if scores[result]==0:

        return "curriculum"



    return result





# ======================================================
# Analyzer
# ======================================================


def analyze_query(question):


    text=normalize_text(question)



    meta=QueryMetadata(

        raw_question=question

    )



    # Grade

    for pattern,value in GRADE_PATTERNS.items():


        if re.search(pattern,text):

            meta.grade=value

            break





    # Subject

    for subject,words in SUBJECT_PATTERNS.items():


        if any(

            w in text

            for w in words

        ):

            meta.subject=subject

            break





    # Term

    for pattern,value in TERM_PATTERNS.items():


        if re.search(pattern,text):

            meta.term=value

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





    # Source Type

    if meta.intent=="news":

        meta.source_category="news"


    elif meta.intent=="exam":

        meta.source_category="exam"





    # Build Query

    query=[


        question,


        meta.subject,


        meta.grade,


        meta.term


    ]



    if meta.intent in [

        "curriculum",

        "summary",

        "explanation",

        "exercise"

    ]:


        query.append(

            "منهج وزارة التربية والتعليم مصر"

        )





    meta.search_query=" ".join(

        x

        for x in query

        if x

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

        f"""

        Query={question}

        Intent={meta.intent}

        Subject={meta.subject}

        Grade={meta.grade}

        Term={meta.term}

        """

    )



    return meta
