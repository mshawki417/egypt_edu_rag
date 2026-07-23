"""
Production Arabic Document Cleaner
Education RAG Compatible Version

Supports:
- clean_document
- clean_text
- is_arabic_heavy
- extract_arabic_sections
"""

from __future__ import annotations


import re
import hashlib
import unicodedata





# =====================================================
# Regex
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


ARABIC_DIACRITICS = re.compile(
    r"[\u064B-\u065F\u0670]"
)


PAGE_PATTERN = re.compile(
    r"\[?\s*صفحة\s*\d+\s*\]?"
)





# =====================================================
# Noise
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

]





EDUCATION_WORDS = [

    "درس",

    "منهج",

    "كتاب",

    "الصف",

    "الترم",

    "الفصل",

    "وزارة التربية",

    "السؤال",

    "الإجابة",

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

        "ئ":"ي"

    }



    for old,new in replacements.items():

        text=text.replace(

            old,

            new

        )



    text=ARABIC_DIACRITICS.sub(

        "",

        text

    )


    return text





# =====================================================
# Noise Removal
# =====================================================


def remove_noise(text:str)->str:


    result=[]


    for line in text.splitlines():


        line=line.strip()


        if not line:

            continue



        remove=False



        for pattern in NOISE_PATTERNS:


            if re.match(

                pattern,

                line,

                re.IGNORECASE

            ):

                remove=True

                break



        if not remove:

            result.append(line)



    return "\n".join(result)






# =====================================================
# Duplicate Handling
# =====================================================


def remove_duplicates(text:str)->str:


    seen=set()

    output=[]



    for line in text.splitlines():


        key=line.strip()



        if not key:

            continue



        if key in seen:

            continue



        seen.add(key)

        output.append(key)



    return "\n".join(output)







def remove_repeated_blocks(text:str)->str:


    paragraphs=text.split("\n\n")


    seen=set()

    output=[]



    for p in paragraphs:


        p=p.strip()


        if not p:

            continue



        h=hashlib.md5(

            p.encode("utf-8")

        ).hexdigest()



        if h not in seen:


            seen.add(h)

            output.append(p)



    return "\n\n".join(output)






# =====================================================
# Arabic Detection
# =====================================================


def is_arabic_heavy(

    text:str,

    threshold:float=0.15

)->bool:


    if not text:

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


    return ratio >= threshold






# =====================================================
# Extract Arabic Sections
# =====================================================


def extract_arabic_sections(

    text:str,

    min_len:int=40

)->str:


    sections=[]



    for line in text.splitlines():


        line=line.strip()


        if len(line)<min_len:

            continue



        if is_arabic_heavy(

            line,

            threshold=0.20

        ):

            sections.append(line)



    return "\n\n".join(sections)






# =====================================================
# Quality
# =====================================================


def quality_score(text:str)->float:


    if not text:

        return 0



    score=0



    length=len(text)



    if length>500:

        score+=0.25



    if length>1500:

        score+=0.15



    if is_arabic_heavy(

        text,

        0.3

    ):

        score+=0.25



    matches=sum(

        1

        for x in EDUCATION_WORDS

        if x in text

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
# Validation
# =====================================================


def is_valid_document(text:str)->bool:


    if len(text)<80:

        return False



    return is_arabic_heavy(

        text,

        0.15

    )






# =====================================================
# Hash
# =====================================================


def document_hash(text:str)->str:


    return hashlib.md5(

        text.encode("utf-8")

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



    text=PAGE_PATTERN.sub(

        " ",

        text

    )



    text=normalize_arabic(

        text

    )



    text=remove_noise(

        text

    )



    text=extract_arabic_sections(

        text

    ) or text



    text=remove_duplicates(

        text

    )



    text=remove_repeated_blocks(

        text

    )



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


    result=clean_document(text)


    content=result.get(

        "content",

        ""

    )


    return content.strip()
