"""
frontend/app.py

Streamlit frontend for the Egypt Education RAG System.
Run with:  streamlit run frontend/app.py
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from loguru import logger

from backend.rag.orchestrator import run_rag_pipeline, get_pipeline_metadata
from backend.rag.chain import RAGAnswer


# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="نظام التعليم المصري الذكي",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── CSS — white professional theme, Arabic-ready ───────────────────────────────
st.markdown("""
<style>
/* ── Google Font for Arabic ─────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700&display=swap');

/* ── Root / Base ─────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Cairo', 'Segoe UI', sans-serif;
    direction: rtl;
}

.main { background-color: #FFFFFF; }
.block-container { padding: 2rem 3rem 2rem 3rem; max-width: 1100px; }

/* ── Header ─────────────────────────────────────── */
.edu-header {
    background: linear-gradient(135deg, #1a3a5c 0%, #2d6a9f 100%);
    color: white;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    text-align: center;
}
.edu-header h1 { font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
.edu-header p  { font-size: 1rem; opacity: 0.85; margin: 0.5rem 0 0; }

/* ── Cards ───────────────────────────────────────── */
.card {
    background: #FFFFFF;
    border: 1px solid #E8EDF2;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* ── Answer box ──────────────────────────────────── */
.answer-box {
    background: #F8FBFF;
    border-right: 4px solid #2d6a9f;
    border-radius: 0 12px 12px 0;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    font-size: 1.05rem;
    line-height: 1.9;
    color: #1a2332;
}

/* ── Source badge ────────────────────────────────── */
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

/* ── Metadata pill ───────────────────────────────── */
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

/* ── Steps ───────────────────────────────────────── */
.step-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 0;
    color: #5a7a9a;
    font-size: 0.9rem;
}
.step-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #2d6a9f;
    flex-shrink: 0;
}

/* ── Streamlit widget overrides ──────────────────── */
div[data-testid="stTextArea"] textarea {
    font-family: 'Cairo', sans-serif !important;
    direction: rtl;
    font-size: 1rem;
    border-radius: 10px;
    border-color: #D0DCE8;
}
div[data-testid="stTextArea"] textarea:focus { border-color: #2d6a9f !important; }

.stButton > button {
    background: linear-gradient(135deg, #1a3a5c, #2d6a9f);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-family: 'Cairo', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    transition: opacity 0.2s;
    width: 100%;
}
.stButton > button:hover { opacity: 0.88; }

.stSelectbox > div { font-family: 'Cairo', sans-serif; }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] { background: #F5F8FC; border-left: 1px solid #E0E8F0; }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

/* ── Divider ─────────────────────────────────────── */
hr { border: none; border-top: 1px solid #EAF0F6; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # list of {question, answer, sources, meta}


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ إعدادات النظام")

    retriever_strategy = st.selectbox(
        "استراتيجية الاسترجاع",
        options=["hybrid", "dense", "bm25"],
        format_func=lambda x: {
            "hybrid": "🔀 هجين (BM25 + Dense) — الأفضل",
            "dense":  "🧠 Dense (دلالي / Semantic)",
            "bm25":   "🔤 BM25 (كلمات مفتاحية)",
        }[x],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 📚 أمثلة على الأسئلة")
    example_questions = [
        "ما هو منهج العلوم للصف الخامس الابتدائي الترم الأول؟",
        "ما موضوعات الرياضيات للصف الثالث الإعدادي؟",
        "ما جدول امتحانات الثانوية العامة ٢٠٢٥؟",
        "ما أهداف تدريس اللغة العربية في المرحلة الابتدائية؟",
        "ما آخر قرارات وزارة التربية والتعليم؟",
    ]
    for q in example_questions:
        if st.button(q, key=f"ex_{q[:20]}", use_container_width=True):
            st.session_state["prefill_question"] = q

    st.markdown("---")
    st.markdown("### ℹ️ عن النظام")
    st.markdown("""
    <div style="font-size:0.82rem;color:#5a7a9a;line-height:1.7">
    يقوم النظام بـ:
    <ul style="padding-right:1rem">
    <li>تحليل سؤالك تلقائيًا</li>
    <li>البحث الحي في مصادر وزارة التربية</li>
    <li>معالجة النصوص العربية</li>
    <li>استرجاع أدق المعلومات</li>
    <li>توليد إجابة مستندة بالمصادر</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ── Main header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="edu-header">
    <h1>🎓 نظام الذكاء الاصطناعي للتعليم المصري</h1>
    <p>اسأل عن المناهج · الامتحانات · القرارات الوزارية · المدارس</p>
</div>
""", unsafe_allow_html=True)


# ── Question input ─────────────────────────────────────────────────────────────
prefill = st.session_state.pop("prefill_question", "")

col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_area(
        label="سؤالك",
        value=prefill,
        placeholder="مثال: ما هو منهج العلوم للصف الخامس الابتدائي الترم الأول؟",
        height=100,
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 ابحث", use_container_width=True)


# ── Run pipeline ───────────────────────────────────────────────────────────────
if search_clicked and question.strip():
    with st.container():
        # Show query metadata
        meta = get_pipeline_metadata(question)
        pills_html = "".join([
            f'<span class="meta-pill">{k}: {v}</span>'
            for k, v in meta.items()
            if v and k not in ("search_query", "needs_live_search")
        ])
        if pills_html:
            st.markdown(f"**تحليل السؤال:** {pills_html}", unsafe_allow_html=True)

        # Pipeline steps progress
        steps_placeholder = st.empty()
        steps = [
            "🔍 تحليل السؤال…",
            "🌐 البحث في مصادر وزارة التربية…",
            "🧹 معالجة النصوص العربية…",
            "📊 بناء فهرس الاسترجاع…",
            "💡 توليد الإجابة…",
        ]

        for i, step_text in enumerate(steps[:4]):
            steps_placeholder.markdown(
                f'<div class="step-indicator"><div class="step-dot"></div>{step_text}</div>',
                unsafe_allow_html=True,
            )

        # Answer streaming
        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        answer_placeholder = st.empty()
        steps_placeholder.markdown(
            f'<div class="step-indicator"><div class="step-dot"></div>{steps[4]}</div>',
            unsafe_allow_html=True,
        )

        full_answer = ""
        try:
            token_stream = run_rag_pipeline(
                question=question,
                retriever_strategy=retriever_strategy,
                stream=True,
            )
            for token in token_stream:
                full_answer += token
                answer_placeholder.markdown(full_answer)

        except Exception as exc:
            full_answer = f"⚠️ حدث خطأ أثناء المعالجة: {exc}"
            answer_placeholder.markdown(full_answer)
            logger.error(f"Pipeline error: {exc}")

        st.markdown('</div>', unsafe_allow_html=True)
        steps_placeholder.empty()

        # Sources (we re-run non-streaming to get source metadata)
        try:
            batch_result: RAGAnswer = run_rag_pipeline(
                question=question,
                retriever_strategy=retriever_strategy,
                stream=False,
            )
            if batch_result.sources:
                st.markdown("**📎 المصادر المستخدمة:**", unsafe_allow_html=True)
                badges = "".join([
                    f'<a class="source-badge" href="{s["url"]}" target="_blank">'
                    f'🔗 {s["title"][:60]}</a>'
                    for s in batch_result.sources
                ])
                st.markdown(badges, unsafe_allow_html=True)

                with st.expander("تفاصيل الاسترجاع"):
                    st.json({
                        "retriever": batch_result.retriever_used,
                        "chunks_retrieved": batch_result.chunks_retrieved,
                        "sources": batch_result.sources,
                    })

            # Save to history
            st.session_state.history.insert(0, {
                "question": question,
                "answer": full_answer,
                "sources": batch_result.sources,
                "meta": meta,
            })

        except Exception:
            pass  # Sources display is best-effort

elif search_clicked and not question.strip():
    st.warning("⚠️ يرجى كتابة سؤالك أولًا.")


# ── Conversation history ───────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕘 الأسئلة السابقة")
    for i, item in enumerate(st.session_state.history[1:6], 1):   # show last 5
        with st.expander(f"{item['question'][:80]}…" if len(item['question']) > 80 else item['question']):
            st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)
            if item.get("sources"):
                badges = "".join([
                    f'<a class="source-badge" href="{s["url"]}" target="_blank">{s["title"][:50]}</a>'
                    for s in item["sources"]
                ])
                st.markdown(badges, unsafe_allow_html=True)

    if st.button("🗑️ مسح السجل", use_container_width=False):
        st.session_state.history = []
        st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<hr>
<div style="text-align:center;color:#8A9AAA;font-size:0.8rem;padding:1rem 0">
    نظام RAG التعليمي المصري · مدعوم بـ Claude AI · المصدر: وزارة التربية والتعليم المصرية
</div>
""", unsafe_allow_html=True)
