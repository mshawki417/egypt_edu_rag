"""
Production Arabic Document Cleaner
Optimized for Education RAG
"""


from __future__ import annotations


import re
import hashlib
import unicodedata




# =========================
# Patterns
# =========================


ARABIC_LETTERS = re.compile(
    r"[\u0600-\u06FF]"
)


URL_PATTERN = re.compile(
    r"https?://\S+"
)


EMAIL_PATTERN = re.compile(
    r"\S+@\S+"
)


HTML_PATTERN = re.compile(
    r"<[^>]+>"
)


SPACES_PATTERN = re.compile(
    r"[ \t]+"
)


NEWLINE_PATTERN = re.compile(
    r"\n{3,}"
)



NOISE = [

    "الرئيسية",

    "تسجيل الدخول",

    "اشترك",

    "تابعنا",

    "اعلان",

    "إعلان",

    "اقرأ المزيد",

    "اضغط هنا",

    "التالي",

    "السابق",

    "سياسة الخصوصية",

    "جميع الحقوق محفوظة",

    "حقوق النشر",

]




# =========================
# Normalize Arabic
# =========================


def normalize_arabic(text):


    text=unicodedata.normalize(
        "NFC",
        text
    )


    replacements={

        "أ":"ا",
        "إ":"ا",
        "آ":"ا",
        "ٱ":"ا",
        "ى":"ي",
        "ؤ":"و",
        "ئ":"ي"

    }


    for a,b in replacements.items():

        text=text.replace(
            a,
            b
        )


    # remove tashkeel

    text=re.sub(

        r"[\u064B-\u065F]",

        "",

        text

    )


    return text





# =========================
# Remove Noise
# =========================


def remove_noise(text):


    lines=[]


    for line in text.splitlines():


        line=line.strip()


        if not line:

            continue



        skip=False


        for word in NOISE:

            if word.lower() in line.lower():

                skip=True

                break



        if not skip:

            lines.append(line)



    return "\n".join(lines)






# =========================
# Duplicate Removal
# =========================


def remove_duplicates(text):


    seen=set()

    result=[]


    for line in text.splitlines():


        key=line.strip()


        if not key:

            continue



        if key in seen:

            continue



        seen.add(key)

        result.append(key)



    return "\n".join(result)






# =========================
# Similarity Filter
# =========================


def remove_repeated_blocks(text):


    paragraphs=text.split("\n\n")


    seen=set()

    output=[]



    for p in paragraphs:


        fingerprint=hashlib.md5(

            p.encode()

        ).hexdigest()



        if fingerprint not in seen:

            seen.add(fingerprint)

            output.append(p)



    return "\n\n".join(output)







# =========================
# Quality Score
# =========================


def quality_score(text):


    score=0



    length=len(text)



    if length>500:

        score+=0.3



    if length>1500:

        score+=0.2




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

        score+=0.3



    if any(

        x in text

        for x in [

            "درس",

            "منهج",

            "وزارة التربية",

            "الصف"

        ]

    ):

        score+=0.2



    return round(
        score,
        2
    )






# =========================
# Main Cleaner
# =========================


def clean_document(text):


    if not text:

        return {

            "content":"",

            "quality":0

        }



    text=HTML_PATTERN.sub(
        " ",
        text
    )


    text=URL_PATTERN.sub(
        " ",
        text
    )


    text=EMAIL_PATTERN.sub(
        " ",
        text
    )



    text=normalize_arabic(
        text
    )


    text=remove_noise(
        text
    )


    text=remove_duplicates(
        text
    )


    text=remove_repeated_blocks(
        text
    )



    text=SPACES_PATTERN.sub(
        " ",
        text
    )


    text=NEWLINE_PATTERN.sub(
        "\n\n",
        text
    )



    score=quality_score(
        text
    )



    return {

        "content":text.strip(),

        "quality":score

    }
