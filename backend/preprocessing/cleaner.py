"""
Production Arabic Document Cleaner
Optimized for Education RAG
"""

from __future__ import annotations

import re
import hashlib
import unicodedata



# =====================================================
# Regex Patterns
# =====================================================


URL_PATTERN = re.compile(
    r"https?://\S+"
)


EMAIL_PATTERN = re.compile(
    r"\S+@\S+"
)


HTML_PATTERN = re.compile(
    r"<[^>]+>"
)


MULTI_SPACE = re.compile(
    r"[ \t]+"
)


MULTI_NEWLINE = re.compile(
    r"\n{3,}"
)


PAGE_MARKER = re.compile(
    r"\[?\s*صفحة\s*\d+\s*\]?"
)


ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670]"
)



# =====================================================
# Web Noise
# =====================================================


NOISE_PATTERNS = [

    r"^الرئيسية$",

    r"^تسجيل الدخول$",

    r"^تابعنا$",

    r"^اشترك$",

    r"^التالي$",

    r"^السابق$",

    r"^اقرأ المزيد$",

    r"^اضغط هنا$",

    r"^سياسة الخصوصية$",

    r"^جميع الحقوق محفوظة$",

    r"^حقوق النشر$",

    r"^إعلان ممول$",

    r"^اعلان ممول$",

]





# =====================================================
# Educational Keywords
# =====================================================


EDU_KEYWORDS = [

    "درس",

    "منهج",

    "كتاب",

    "الصف",

    "الفصل",

    "الترم",

    "وزارة التربية",

    "السؤال",

    "الإجابة",

    "شرح",

]





# =====================================================
# Arabic Normalize
# =====================================================


def normalize_arabic(text:str)->str:


    text = unicodedata.normalize(

        "NFC",

        text

    )



    replacements = {


        "أ":"ا",

        "إ":"ا",

        "آ":"ا",

        "ٱ":"ا",

        "ى":"ي",

        "ؤ":"و",

        "ئ":"ي",


    }



    for old,new in replacements.items():

        text=text.replace(

            old,

            new

        )



    text = ARABIC_DIACRITICS.sub(

        "",

        text

    )


    return text






# =====================================================
# Remove Web Noise
# =====================================================


def remove_noise(text:str)->str:


    output=[]


    for line in text.splitlines():


        line=line.strip()


        if not line:

            continue



        remove=False



        for pattern in NOISE_PATTERNS:


            if re.search(

                pattern,

                line,

                flags=re.IGNORECASE

            ):

                remove=True

                break



        if not remove:

            output.append(line)



    return "\n".join(output)







# =====================================================
# Normalize Lines
# =====================================================


def normalize_lines(text:str)->str:


    lines=[]


    for line in text.splitlines():


        line=line.strip()


        if len(line)<3:

            continue


        lines.append(line)



    return "\n".join(lines)








# =====================================================
# Remove Duplicate Lines
# =====================================================


def remove_duplicates(text:str)->str:


    seen=set()

    result=[]



    for line in text.splitlines():


        key=line.strip()



        if key in seen:

            continue



        seen.add(key)

        result.append(key)



    return "\n".join(result)








# =====================================================
# Remove Duplicate Paragraphs
# =====================================================


def remove_repeated_blocks(text:str)->str:


    paragraphs=text.split("\n\n")


    seen=set()

    output=[]



    for paragraph in paragraphs:


        paragraph=paragraph.strip()


        if not paragraph:

            continue



        fingerprint=hashlib.md5(

            paragraph.encode(

                "utf-8"

            )

        ).hexdigest()



        if fingerprint not in seen:


            seen.add(fingerprint)

            output.append(paragraph)



    return "\n\n".join(output)








# =====================================================
# Quality Detection
# =====================================================


def quality_score(text:str)->float:


    if not text:

        return 0



    score=0



    length=len(text)



    # Size

    if length>500:

        score+=0.25


    if length>1500:

        score+=0.15



    # Arabic Ratio

    arabic=sum(

        1

        for c in text

        if "\u0600" <= c <= "\u06FF"

    )



    ratio=arabic/max(

        length,

        1

    )



    if ratio>0.3:

        score+=0.25



    # Education relevance

    matches=sum(

        1

        for word in EDU_KEYWORDS

        if word in text

    )



    score+=min(

        matches*0.05,

        0.25

    )



    return round(

        min(score,1),

        2

    )








# =====================================================
# Document Validation
# =====================================================


def is_valid_document(text:str)->bool:


    if len(text)<80:

        return False



    arabic=sum(

        1

        for c in text

        if "\u0600" <= c <= "\u06FF"

    )



    ratio=arabic/max(

        len(text),

        1

    )


    return ratio>=0.15








# =====================================================
# Hash
# =====================================================


def document_hash(text:str)->str:


    return hashlib.md5(

        text.encode(

            "utf-8"

        )

    ).hexdigest()








# =====================================================
# Main Cleaner
# =====================================================


def clean_document(text:str)->dict:


    if not text:


        return {

            "content":"",

            "quality":0,

            "hash":None

        }



    # HTML

    text=HTML_PATTERN.sub(

        " ",

        text

    )



    # URLs

    text=URL_PATTERN.sub(

        " ",

        text

    )



    # Emails

    text=EMAIL_PATTERN.sub(

        " ",

        text

    )



    # PDF pages

    text=PAGE_MARKER.sub(

        " ",

        text

    )



    # Arabic normalize

    text=normalize_arabic(

        text

    )



    # Noise

    text=remove_noise(

        text

    )



    text=normalize_lines(

        text

    )



    text=remove_duplicates(

        text

    )



    text=remove_repeated_blocks(

        text

    )



    # Spaces

    text=MULTI_SPACE.sub(

        " ",

        text

    )


    text=MULTI_NEWLINE.sub(

        "\n\n",

        text

    )



    text=text.strip()



    if not is_valid_document(text):


        return {

            "content":"",

            "quality":0,

            "hash":None

        }



    return {


        "content":text,


        "quality":quality_score(text),


        "hash":document_hash(text)

    }









# =====================================================
# Backward Compatibility
# =====================================================


def clean_text(

    text:str,

    normalize:bool=True

)->str:


    """
    Compatibility function.

    Used by old modules:
    - chunker
    - scraper
    - preprocessing
    """


    result=clean_document(

        text

    )



    content=result.get(

        "content",

        ""

    )



    if not normalize:

        return content



    return content.strip()
