"""
Advanced Arabic Text Cleaner
Optimized for Real-Time RAG
"""


from __future__ import annotations


import re
import unicodedata



# ==========================
# Regex
# ==========================


ALEF_VARIANTS = re.compile(
    r"[أإآٱ]"
)


DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670]"
)


HTML_TAGS = re.compile(
    r"<[^>]+>"
)


URL_PATTERN = re.compile(
    r"https?://\S+"
)


EMAIL_PATTERN = re.compile(
    r"\S+@\S+"
)


MULTI_SPACE = re.compile(
    r"[ \t]+"
)


MULTI_NEWLINE = re.compile(
    r"\n{3,}"
)


PAGE_MARKERS = re.compile(
    r"\[صفحة\s*\d+\]"
)



# Common web noise

NOISE_PATTERNS=[


    r"اقرا المزيد",


    r"اقرأ المزيد",


    r"اضغط هنا",


    r"تسجيل الدخول",


    r"حقوق النشر",


    r"جميع الحقوق محفوظة",


    r"سياسة الخصوصية",

]





# ==========================
# Arabic Normalize
# ==========================


def normalize_arabic(
    text:str
):


    text = unicodedata.normalize(
        "NFC",
        text
    )


    # unify alef

    text = ALEF_VARIANTS.sub(
        "ا",
        text
    )


    # remove tashkeel

    text = DIACRITICS.sub(
        "",
        text
    )


    return text





# ==========================
# Remove Noise
# ==========================


def remove_noise(
    text:str
):


    for pattern in NOISE_PATTERNS:


        text=re.sub(

            pattern,

            " ",

            text,

            flags=re.IGNORECASE

        )


    return text





# ==========================
# Main Cleaner
# ==========================


def clean_text(

    text:str,

    normalize=True

):


    if not text:

        return ""



    # Unicode

    text=unicodedata.normalize(
        "NFC",
        text
    )



    # HTML

    text=HTML_TAGS.sub(
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

    text=PAGE_MARKERS.sub(
        "",
        text
    )



    # Noise

    text=remove_noise(
        text
    )



    if normalize:

        text=normalize_arabic(
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



    return text.strip()





# ==========================
# Arabic Ratio
# ==========================


def is_arabic_heavy(

    text:str,

    threshold:float=0.15

):


    if not text:

        return False



    arabic=sum(

        1

        for c in text

        if "\u0600" <= c <= "\u06FF"

    )



    return (

        arabic /
        len(text)

    ) >= threshold





# ==========================
# Extract Arabic Content
# ==========================


def extract_arabic_sections(

    text:str,

    min_len:int=40

):


    result=[]



    for line in text.splitlines():


        line=line.strip()


        if len(line)<min_len:

            continue



        ratio=sum(

            1

            for c in line

            if "\u0600" <= c <= "\u06FF"

        ) / max(
            len(line),
            1
        )



        if ratio >=0.20:

            result.append(
                line
            )



    return "\n\n".join(
        result
    )





# ==========================
# Deduplicate
# ==========================


def remove_duplicate_lines(

    text:str

):


    seen=set()

    output=[]


    for line in text.splitlines():


        clean=line.strip()


        if clean and clean not in seen:

            seen.add(clean)

            output.append(clean)



    return "\n".join(
        output
    )
