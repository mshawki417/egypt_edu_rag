# 🇪🇬 Egypt Edu RAG

نظام **RAG (Retrieval-Augmented Generation)** لمساعدة طلاب المدارس المصرية على فهم المنهج المصري، بيجمع بين البحث في مصادر تعليمية مصرية موثوقة والبحث الحي (Real-Time) على الإنترنت عشان يديك إجابات دقيقة ومحدّثة.

A **Retrieval-Augmented Generation (RAG)** system that helps Egyptian students understand their national curriculum by combining trusted educational sources with real-time web search for accurate, up-to-date answers.

🔗 **Live Demo:** [egyptedurag-2026.streamlit.app](https://egyptedurag-2026.streamlit.app/)

---

## 📋 المحتوى | Table of Contents

- [نظرة عامة | Overview](#-نظرة-عامة--overview)
- [المميزات | Features](#-المميزات--features)
- [كيف يشتغل | How It Works](#-كيف-يشتغل--how-it-works)
- [التثبيت | Installation](#-التثبيت--installation)
- [الإعداد | Configuration](#-الإعداد--configuration)
- [الاستخدام | Usage](#-الاستخدام--usage)
- [أمثلة | Examples](#-أمثلة--examples)
- [هيكل المشروع | Project Structure](#-هيكل-المشروع--project-structure)
- [استكشاف الأخطاء | Troubleshooting](#-استكشاف-الأخطاء--troubleshooting)

---

## 🎯 نظرة عامة | Overview

**بالعربي:**
المشروع ده RAG pipeline مبني بلغة Python وواجهة Streamlit، بيساعد الطلاب وأولياء الأمور والمدرسين يلاقوا إجابات على أسئلة المنهج المصري (لغة عربية، تاريخ، علوم، إلخ) من خلال:
1. استرجاع معلومات من مصادر تعليمية مصرية موثوقة (وزارة التربية والتعليم، بنك المعرفة المصري، ويكيبيديا، إلخ)
2. بحث حي (Live Scraping) لجلب أحدث المعلومات وقت السؤال
3. توليد إجابة دقيقة باستخدام نموذج لغوي عبر OpenRouter

**English:**
This project is a Python-based RAG pipeline with a Streamlit frontend that helps students, parents, and teachers get answers about the Egyptian curriculum (Arabic language, history, science, etc.) by:
1. Retrieving information from trusted Egyptian educational sources (Ministry of Education, Egyptian Knowledge Bank, Wikipedia, etc.)
2. Performing live scraping to fetch the most current information at query time
3. Generating accurate answers using an LLM via OpenRouter

---

## ✨ المميزات | Features

- 🔍 **بحث هجين (Hybrid Search):** BM25 + FAISS embeddings للحصول على أدق النتائج
- 🎯 **إعادة ترتيب النتائج (Reranking):** لتحديد أكتر 4 مقاطع (chunks) ملائمة للسؤال
- 🧠 **تحليل نية السؤال (Query Analysis):** تحديد الـ Intent (منهج / امتحان...) والمادة والصف الدراسي تلقائيًا
- 🌐 **Live Scraping:** بحث حي عبر DuckDuckGo مقيد بمصادر مصرية موثوقة فقط
- 📄 **دعم PDF و HTML:** استخراج نصوص من صفحات الويب وملفات PDF
- ⚡ **Caching ذكي:** تخزين مؤقت (TTL Cache) لتقليل وقت الاستجابة على الأسئلة المتكررة
- 🤖 **نماذج مجانية عبر OpenRouter:** يدعم `openrouter/free` auto-router لتفادي انقطاع الخدمة عند توقف موديل معين
- 🚦 **تحكم في التزامن (Concurrency Control):** Semaphore مربوط بالـ event loop لتفادي أخطاء asyncio في بيئة Streamlit

---

## ⚙️ كيف يشتغل | How It Works

```
سؤال المستخدم
      │
      ▼
┌─────────────────┐
│ Query Analyzer   │  → تحديد الـ Intent / Subject / Grade
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Retrieval        │  → BM25 + FAISS على المستندات المفهرسة
│ (Hybrid Search)  │
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Reranking        │  → اختيار أفضل 4 مقاطع
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Live Scraping    │  → بحث حي DDG على مصادر موثوقة (اختياري)
└────────┬─────────┘
         │
         ▼
┌─────────────────┐
│ Answer Generation │  → OpenRouter LLM
│ (OpenRouter)     │
└────────┬─────────┘
         │
         ▼
     الإجابة النهائية
```

**User query → Query Analysis → Hybrid Retrieval (BM25 + FAISS) → Reranking → Live Scraping (optional) → Answer Generation via OpenRouter → Final Answer**

---

## 🛠️ التثبيت | Installation

### المتطلبات | Prerequisites
- Python 3.10+
- مفتاح API من [OpenRouter](https://openrouter.ai/) (مجاني)

### الخطوات | Steps

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/egypt-edu-rag.git
cd egypt-edu-rag

# 2. إنشاء بيئة افتراضية (Virtual Environment)
python -m venv venv
source venv/bin/activate  # على ويندوز: venv\Scripts\activate

# 3. تثبيت المكتبات
pip install -r requirements.txt

# 4. تشغيل المشروع
streamlit run frontend/app.py
```

---

## 🔐 الإعداد | Configuration

اعمل ملف `.env` في جذر المشروع وحط فيه:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=openrouter/free
```

> **ملحوظة:** الموديلات المجانية على OpenRouter بتتغير باستمرار من غير إشعار مسبق، فاستخدام `openrouter/free` (auto-router) بيضمن إن الـ pipeline يفضل شغال حتى لو موديل معين اتشال أو وصل للـ rate limit.
>
> **Note:** Free models on OpenRouter change frequently without notice. Using `openrouter/free` (the auto-router) ensures the pipeline keeps working even if a specific model is deprecated or rate-limited.

---

## 🚀 الاستخدام | Usage

1. افتح المشروع على المتصفح (محليًا على `http://localhost:8501` أو عبر [الرابط المباشر](https://egyptedurag-2026.streamlit.app/))
2. اكتب سؤالك بالعربي في مربع البحث (مثال: "ملخص الدرس الأول عربي الصف الثالث")
3. المشروع هيحلل السؤال، يدور في المصادر المفهرسة، ويعمل بحث حي لو محتاج، وبعدين يديك إجابة مبنية على مصادر موثوقة

---

## 💡 أمثلة | Examples

| السؤال (Query) | الـ Intent المتوقع | المادة (Subject) |
|---|---|---|
| ملخص الدرس الأول عربي الصف الثالث | `curriculum` | لغة عربية |
| أسئلة امتحان تاريخ محمد علي باشا | `exam` | تاريخ |
| اشرحلي درس الكسور رياضيات ابتدائي | `curriculum` | رياضيات |

### مثال Python للاستخدام البرمجي | Programmatic Usage Example

```python
from backend.rag.orchestrator import run_rag_pipeline

result = run_rag_pipeline(
    query="ملخص الدرس الاول عربي الصف الثالث"
)

print(result["answer"])
print(result["sources"])
```

---

## 📁 هيكل المشروع | Project Structure

```
egypt-edu-rag/
├── frontend/
│   └── app.py                  # واجهة Streamlit
├── backend/
│   ├── rag/
│   │   ├── orchestrator.py     # تنسيق كامل الـ pipeline
│   │   └── chain.py            # استدعاء OpenRouter API
│   ├── retrieval/
│   │   └── retriever.py        # BM25 + FAISS indexing/search
│   └── scraper/
│       ├── live_scraper.py     # البحث والاستخراج الحي
│       └── query_analyzer.py   # تحليل نية السؤال
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🔧 استكشاف الأخطاء | Troubleshooting

| المشكلة | السبب | الحل |
|---|---|---|
| `404 No endpoints found` من OpenRouter | الموديل المحدد اتشال أو مش متاح مجانًا | استخدم `openrouter/free` بدل تحديد موديل معين |
| `Semaphore is bound to a different event loop` | Streamlit بيعمل event loop جديد كل rerun | تأكد إن الـ Semaphore متعمل جوه الـ event loop الحالي (مش على مستوى الـ module) |
| مفيش نتائج من الـ Live Scraping | الدومينز المسموحة (`ALLOWED_DOMAINS`) محدودة | راجع/زوّد قايمة الدومينز في `live_scraper.py` |

---

## 📄 الترخيص | License

هتحدده انت حسب احتياجك (MIT / Apache 2.0 / إلخ) — لو مش متأكد، **MIT License** خيار شائع وبسيط للمشاريع مفتوحة المصدر.

---

## 🤝 المساهمة | Contributing

المساهمات مرحّب بيها! افتح Issue أو ابعت Pull Request.

Contributions are welcome! Feel free to open an issue or submit a pull request.
