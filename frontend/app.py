"""
frontend/app.py
Egypt Education RAG — Streamlit frontend
Fixed: CSS RTL, pipeline runs once (not twice), secrets reading
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from loguru import logger

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="نظام التعليم المصري الذكي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — white professional, RTL fixed ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

/* Base — apply RTL only to content, NOT to Streamlit widgets */
.main .block-container {
    font-family: 'Cairo', 'Segoe UI', sans-serif;
    direction: rtl;
    padding: 2rem 3rem;
    max-width: 1100px;
}

/* Sidebar RTL */
[data-testid="stSidebar"] {
    font-family: 'Cairo', sans-serif;
    background: #F5F8FC;
    border-left: 1px solid #E0E8F0;
}
[data-testid="stSidebar"] .block-container {
    direction: rtl;
    padding: 1.5rem 1rem;
}

/* Widgets: keep LTR for correct rendering */
div[data-testid="stSelectbox"],
div[data-testid="stTextArea"],
div[data-testid="stButton"] {
    direction: ltr;
}
/* But text inside textarea goes RTL */
div[data-testid="stTextArea"] textarea {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
    font-size: 1rem;
    border-radius: 10px;
}

.main { background: #FFFFFF; }

/* Header */
.edu-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}
.edu-header h1 { font-size: 1.9rem; font-weight: 700; margin: 0; }
.edu-header p  { font-size: 0.95rem; opacity: 0.85; margin: 0.4rem 0 0; }

/* Cards */
.card {
    background: #FFF;
    border: 1px solid #E8EDF2;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* Answer */
.answer-box {
    background: #F8FBFF;
    border-right: 4px solid #2d6a9f;
    border-radius: 0 12px 12px 0;
    padding: 1.4rem 1.8rem;
    margin: 1rem 0;
    font-size: 1.05rem;
    line-height: 1.9;
    color: #1a2332;
    direction: rtl;
}

/* Badges */
.meta-pill {
    display: inline-block;
    background: #F0F4F8;
    color: #3a5a7c;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    font-size: 0.78rem;
    margin: 0.15rem;
    border: 1px solid #D8E4EE;
}
.source-badge {
    display: inline-block;
    background: #EEF4FB;
    color: #1a3a5c;
    border: 1px solid #C5D9EE;
    border-radius: 6px;
    padding: 0.25rem 0.75rem;
    font-size: 0.8rem;
    margin: 0.2rem;
    text-decoration: none;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1a3a5c, #2d6a9f);
    color: white;
    border: none;
    border-radius: 10px;
    font-family: 'Cairo', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

hr { border: none; border-top: 1px solid #EAF0F6; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ إعدادات النظام")
    retriever_strategy = st.selectbox(
        "استراتيجية الاسترجاع",
        options=["hybrid", "dense", "bm25"],
        format_func=lambda x: {
            "hybrid": "🔀 هجين — الأفضل",
            "dense":  "🧠 Dense (دلالي)",
            "bm25":   "🔤 BM25 (كلمات مفتاحية)",
        }[x],
    )

    # API Key status
    st.markdown("---")
    try:
        from config.settings import llm_cfg
        if llm_cfg.openrouter_api_key:
            st.success("✅ OpenRouter API مكوَّن")
        else:
            st.error("❌ OPENROUTER_API_KEY غير موجود")
            st.caption("أضفه في Streamlit Secrets")
    except Exception:
        pass

    st.markdown("---")
    st.markdown("### 📚 أمثلة")
    examples = [
        "ما هو منهج العلوم للصف الخامس الابتدائي الترم الأول؟",
        "ما موضوعات الرياضيات للصف الثالث الإعدادي؟",
        "ما جدول امتحانات الثانوية العامة؟",
        "ما آخر قرارات وزارة التربية والتعليم؟",
        "ما أهداف تدريس اللغة العربية في الابتدائي؟",
    ]
    for q in examples:
        if st.button(q[:45] + "…" if len(q) > 45 else q, key=f"ex_{q[:15]}"):
            st.session_state["prefill_q"] = q

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.8rem;color:#5a7a9a;line-height:1.8;direction:rtl">
    <b>النظام يقوم بـ:</b><br>
    🔍 تحليل السؤال تلقائياً<br>
    🌐 البحث الحي في مواقع الوزارة<br>
    🧹 معالجة النصوص العربية<br>
    📊 استرجاع بـ 3 استراتيجيات<br>
    💡 توليد إجابة مستندة بالمصادر
    </div>
    """, unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-header">
    <h1>🎓 نظام الذكاء الاصطناعي للتعليم المصري</h1>
    <p>اسأل عن المناهج · الامتحانات · القرارات الوزارية · المدارس</p>
</div>
""", unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_q", "")
col1, col2 = st.columns([5, 1])
with col1:
    question = st.text_area(
        "سؤالك",
        value=prefill,
        placeholder="مثال: ما هو منهج العلوم للصف الخامس الابتدائي الترم الأول؟",
        height=100,
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    search_btn = st.button("🔍 ابحث", use_container_width=True)

# ── Run pipeline ───────────────────────────────────────────────────────────────
if search_btn and question.strip():
    # Show metadata
    try:
        from backend.rag.orchestrator import get_pipeline_metadata
        meta = get_pipeline_metadata(question)
        st.session_state.last_meta = meta
        pills = "".join([
            f'<span class="meta-pill">{k}: {v}</span>'
            for k, v in meta.items()
            if v and v != "—" and k not in ("search_query", "needs_live_search")
        ])
        if pills:
            st.markdown(f"**تحليل السؤال:** {pills}", unsafe_allow_html=True)
    except Exception:
        pass

    # Pipeline steps
    status = st.empty()
    steps = ["🔍 تحليل السؤال…", "🌐 جلب البيانات من الوزارة…",
             "🧹 معالجة النصوص…", "📊 فهرسة واسترجاع…", "💡 توليد الإجابة…"]

    for step in steps[:4]:
        status.info(step)

    # Stream answer — pipeline runs ONCE only
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    answer_ph = st.empty()
    status.info(steps[4])

    full_answer = ""
    retrieved_sources = []

    try:
        from backend.rag.orchestrator import run_rag_pipeline
        from backend.rag.chain import RAGAnswer

        # Run once in batch mode — faster and avoids double-call
        result: RAGAnswer = run_rag_pipeline(
            question=question,
            retriever_strategy=retriever_strategy,
            stream=False,
        )
        full_answer = result.answer
        retrieved_sources = result.sources
        answer_ph.markdown(full_answer)

    except Exception as exc:
        full_answer = f"⚠️ حدث خطأ: {exc}"
        answer_ph.markdown(full_answer)
        logger.error(f"Pipeline error: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)
    status.empty()

    # Sources
    st.session_state.last_sources = retrieved_sources
    if retrieved_sources:
        st.markdown("**📎 المصادر:**", unsafe_allow_html=True)
        badges = "".join([
            f'<a class="source-badge" href="{s["url"]}" target="_blank">'
            f'🔗 {s["title"][:55]}</a>'
            for s in retrieved_sources
        ])
        st.markdown(badges, unsafe_allow_html=True)

        with st.expander("📊 تفاصيل الاسترجاع"):
            st.json({
                "retriever": retrieved_sources[0].get("retriever", "—") if retrieved_sources else "—",
                "chunks": len(retrieved_sources),
                "sources": retrieved_sources,
            })

    # Save history
    st.session_state.history.insert(0, {
        "question": question,
        "answer": full_answer,
        "sources": retrieved_sources,
    })

elif search_btn and not question.strip():
    st.warning("⚠️ يرجى كتابة سؤالك أولاً.")

# ── History ────────────────────────────────────────────────────────────────────
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.markdown("### 🕘 الأسئلة السابقة")
    for item in st.session_state.history[1:6]:
        label = item["question"][:70] + "…" if len(item["question"]) > 70 else item["question"]
        with st.expander(label):
            st.markdown(
                f'<div class="answer-box">{item["answer"]}</div>',
                unsafe_allow_html=True,
            )
            if item.get("sources"):
                badges = "".join([
                    f'<a class="source-badge" href="{s["url"]}" target="_blank">'
                    f'{s["title"][:45]}</a>'
                    for s in item["sources"]
                ])
                st.markdown(badges, unsafe_allow_html=True)

    if st.button("🗑️ مسح السجل"):
        st.session_state.history = []
        st.rerun()

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#8A9AAA;font-size:0.8rem;padding:0.8rem 0;direction:rtl">
    نظام RAG التعليمي المصري · OpenRouter Free LLM · مصدر البيانات: وزارة التربية والتعليم
</div>
""", unsafe_allow_html=True)
