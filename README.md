# 🎓 Egypt Education RAG System
### نظام الذكاء الاصطناعي للتعليم المصري

A production-ready **Retrieval-Augmented Generation (RAG)** system for Egyptian education — live web scraping from Ministry of Education sources, multi-strategy Arabic retrieval, and a clean Streamlit frontend.

**100% free to run** — uses OpenRouter free LLM tier + local embeddings.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
Query Analyzer  ──→  extracts: grade / subject / term / domain
     │
     ▼
Live Scraper    ──→  MOE websites + PDFs (httpx + BeautifulSoup + PyMuPDF)
     │
     ▼
Preprocessor    ──→  Arabic normalization + paragraph-aware chunking
     │
     ▼
Multi-Retriever ──→  BM25  |  Dense (FAISS)  |  Hybrid RRF  ← you choose
     │
     ▼
OpenRouter LLM  ──→  FREE: Llama 3.1 / Qwen2 / Mistral / Gemma
     │
     ▼
Answer  (Arabic · cited · streamed)
```

---

## 📁 Project Structure

```
egypt_edu_rag/
├── backend/
│   ├── scraper/
│   │   ├── query_analyzer.py   # Arabic query parsing → grade/subject/term
│   │   └── live_scraper.py     # httpx scraper with retry + PDF extraction
│   ├── preprocessing/
│   │   ├── cleaner.py          # Arabic normalization (alef, harakat, whitespace)
│   │   └── chunker.py          # Paragraph-aware chunking with overlap
│   ├── retrieval/
│   │   └── retriever.py        # BM25 · Dense (FAISS) · Hybrid (RRF)
│   └── rag/
│       ├── chain.py            # OpenRouter API call + SSE streaming
│       └── orchestrator.py     # Full pipeline wiring
├── frontend/
│   └── app.py                  # Streamlit UI — white, RTL, professional
├── config/
│   └── settings.py             # All config in one place (pydantic)
├── tests/
│   └── test_pipeline.py        # Pytest unit tests
├── .streamlit/
│   ├── config.toml             # Theme: white + blue
│   └── secrets.toml.example    # Secrets template
├── .env.example                # Local dev env template
├── requirements.txt
└── .github/workflows/ci.yml    # GitHub Actions CI
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/YOUR_USERNAME/egypt-edu-rag.git
cd egypt-edu-rag
pip install -r requirements.txt
```

### 2. Get a Free OpenRouter Key
1. Go to [openrouter.ai/keys](https://openrouter.ai/keys)
2. Sign up (free) → Create key
3. Free models available immediately — no credit card needed

### 3. Configure
```bash
cp .env.example .env
# Edit .env → set OPENROUTER_API_KEY=sk-or-v1-...
```

### 4. Run
```bash
streamlit run frontend/app.py
```

---

## 🆓 Free LLM Models (OpenRouter)

Set `LLM_MODEL` in `.env` to any of these:

| Model | Speed | Arabic Quality |
|-------|-------|---------------|
| `meta-llama/llama-3.1-8b-instruct:free` | ⚡ Fast | ✅ Good |
| `qwen/qwen-2-7b-instruct:free` | ⚡ Fast | ✅✅ Best |
| `mistralai/mistral-7b-instruct:free` | ⚡ Fast | ✅ Good |
| `google/gemma-2-9b-it:free` | Medium | ✅ Good |

---

## ⚙️ Retrieval Strategies

| Strategy | How it works | Best for |
|----------|-------------|---------|
| **BM25** | Keyword matching (TF-IDF) | Exact terms, names, grades |
| **Dense** | Sentence embeddings + FAISS | Semantic / paraphrased questions |
| **Hybrid** ⭐ | BM25 + Dense via RRF fusion | Best overall — default |

---

## 🌐 Deploy to Streamlit Cloud (Free)

```
1. Push repo to GitHub
2. Go to share.streamlit.io → New app
3. Repo: your-repo  |  Branch: main  |  File: frontend/app.py
4. Advanced → Secrets:
       OPENROUTER_API_KEY = "sk-or-v1-..."
       LLM_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
5. Deploy ✅
```

---

## 🧪 Run Tests
```bash
pytest tests/ -v
```

---

## 📄 License
MIT
