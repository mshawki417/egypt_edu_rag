"""
frontend/app.py
EduSearch Egypt — AI Research Assistant
Premium design: Corporate Modern + Glassmorphism
"""
from __future__ import annotations
import sys, os, time, threading, queue
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
from loguru import logger

from backend.rag.orchestrator import run_rag_pipeline, get_pipeline_metadata
from backend.rag.chain import RAGAnswer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EduSearch Egypt",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Full CSS — matching DESIGN.md ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500;600&family=Cairo:wght@400;600;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ──────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }

.main { background: #fbf8ff; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Layout Shell ──────────────────────────────────── */
.app-shell {
    display: grid;
    grid-template-columns: 260px 1fr 320px;
    min-height: 100vh;
    background: #fbf8ff;
    font-family: 'Inter', sans-serif;
}

/* ── LEFT SIDEBAR ──────────────────────────────────── */
.sidebar-left {
    background: #ffffff;
    border-right: 1px solid #e3e1eb;
    padding: 24px 16px;
    display: flex;
    flex-direction: column;
    gap: 8px;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
}
.brand-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px 20px;
    border-bottom: 1px solid #eeedf7;
    margin-bottom: 8px;
}
.brand-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, #1e40af, #3755c3);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; color: white;
}
.brand-text { line-height: 1.2; }
.brand-name {
    font-family: 'Sora', sans-serif;
    font-size: 14px; font-weight: 700;
    color: #00288e; letter-spacing: -0.01em;
}
.brand-sub {
    font-size: 10px; font-weight: 600;
    color: #757684; letter-spacing: 0.05em;
    text-transform: uppercase;
}
.new-research-btn {
    background: linear-gradient(135deg, #1e40af, #3755c3);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    font-family: 'Inter', sans-serif;
    font-size: 14px; font-weight: 600;
    cursor: pointer;
    width: 100%;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
    justify-content: center;
    transition: opacity 0.2s;
    text-decoration: none;
}
.new-research-btn:hover { opacity: 0.88; }

.nav-section-label {
    font-size: 11px; font-weight: 600;
    color: #757684; letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 12px 12px 4px;
}
.nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 12px;
    border-radius: 8px;
    font-size: 14px; color: #444653;
    cursor: pointer;
    transition: background 0.15s;
    text-decoration: none;
}
.nav-item:hover { background: #f4f2fc; color: #00288e; }
.nav-item.active { background: #eeedf7; color: #00288e; font-weight: 600; }
.nav-item-icon { font-size: 16px; width: 20px; text-align: center; }

.sidebar-footer {
    margin-top: auto;
    border-top: 1px solid #eeedf7;
    padding-top: 16px;
    display: flex; flex-direction: column; gap: 4px;
}

/* ── MAIN CENTER ───────────────────────────────────── */
.main-center {
    padding: 40px 48px;
    display: flex;
    flex-direction: column;
    gap: 24px;
    overflow-y: auto;
    max-height: 100vh;
}

/* Verified badge */
.verified-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(22, 163, 74, 0.1);
    border: 1px solid rgba(22, 163, 74, 0.3);
    color: #15803d;
    border-radius: 9999px;
    padding: 6px 16px;
    font-size: 12px; font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    width: fit-content;
    margin: 0 auto 16px;
}

.hero-section {
    text-align: center;
    padding: 32px 0 16px;
}
.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 32px; font-weight: 700;
    color: #1a1b22;
    letter-spacing: -0.01em;
    line-height: 40px;
    margin: 0 0 8px;
}
.hero-sub {
    font-size: 16px; color: #444653;
    line-height: 24px; margin: 0;
}

/* Search box */
.search-container {
    background: #ffffff;
    border: 1px solid #e3e1eb;
    border-radius: 16px;
    padding: 20px 20px 16px;
    box-shadow: 0 4px 12px rgba(30,64,175,0.05);
}

/* Quick chips */
.try-asking {
    display: flex; align-items: center; gap: 12px;
    flex-wrap: wrap;
    margin-top: 4px;
}
.try-label {
    font-size: 13px; font-weight: 600;
    color: #757684; letter-spacing: 0.02em;
    text-transform: uppercase;
    white-space: nowrap;
}
.try-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #ffffff;
    border: 1px solid #c4c5d5;
    border-radius: 9999px;
    padding: 6px 14px;
    font-size: 13px; color: #444653;
    cursor: pointer;
    transition: all 0.15s;
    white-space: nowrap;
}
.try-chip:hover { border-color: #3755c3; color: #00288e; background: #f4f2fc; }

/* Query metadata pills */
.meta-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.meta-pill {
    display: inline-flex; align-items: center; gap: 4px;
    background: rgba(55,85,195,0.08);
    color: #173bab;
    border-radius: 9999px;
    padding: 3px 10px;
    font-size: 12px; font-weight: 500;
}

/* Answer area */
.answer-glass {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(196,197,213,0.5);
    border-top: 3px solid #3755c3;
    border-radius: 12px;
    padding: 24px 28px;
    font-size: 16px; line-height: 28px;
    color: #1a1b22;
    min-height: 80px;
}

/* Sources */
.sources-row { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
.source-chip {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f4f2fc;
    border: 1px solid #c4c5d5;
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px; color: #173bab;
    text-decoration: none;
    transition: all 0.15s;
}
.source-chip:hover { background: #eeedf7; border-color: #3755c3; }

/* History items */
.history-item {
    background: #ffffff;
    border: 1px solid #e3e1eb;
    border-radius: 12px;
    padding: 16px 20px;
    cursor: pointer;
    transition: box-shadow 0.15s;
}
.history-item:hover { box-shadow: 0 4px 12px rgba(30,64,175,0.08); }
.history-q { font-size: 14px; font-weight: 600; color: #1a1b22; margin-bottom: 4px; }
.history-a { font-size: 13px; color: #757684; overflow: hidden;
              display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 40px 0;
    color: #757684;
    font-size: 15px;
}

/* ── RIGHT SIDEBAR ──────────────────────────────────── */
.sidebar-right {
    background: #ffffff;
    border-left: 1px solid #e3e1eb;
    padding: 24px 20px;
    position: sticky;
    top: 0;
    height: 100vh;
    overflow-y: auto;
}
.processing-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px;
}
.processing-title {
    display: flex; align-items: center; gap: 8px;
    font-family: 'Sora', sans-serif;
    font-size: 16px; font-weight: 600; color: #1a1b22;
}
.live-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #16a34a;
    animation: pulse-green 2s infinite;
}
.live-dot.idle { background: #c4c5d5; animation: none; }
.live-dot.active { background: #f59e0b; animation: pulse-amber 1s infinite; }
@keyframes pulse-green {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.3); }
}
@keyframes pulse-amber {
    0%,100% { opacity: 1; } 50% { opacity: 0.4; }
}

/* Processing steps */
.step-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 12px 0;
    border-bottom: 1px solid #eeedf7;
    position: relative;
}
.step-row:last-child { border-bottom: none; }
.step-icon-wrap {
    width: 28px; height: 28px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 2px;
    font-size: 13px;
    transition: all 0.3s;
}
.step-icon-wrap.pending { background: #eeedf7; color: #757684; }
.step-icon-wrap.active {
    background: rgba(245,158,11,0.15);
    color: #f59e0b;
    box-shadow: 0 0 0 4px rgba(245,158,11,0.1);
    animation: spin-glow 1s infinite linear;
}
.step-icon-wrap.done { background: rgba(22,163,74,0.12); color: #16a34a; }
.step-icon-wrap.error { background: rgba(186,26,26,0.1); color: #ba1a1a; }
@keyframes spin-glow {
    0% { box-shadow: 0 0 0 2px rgba(245,158,11,0.2); }
    50% { box-shadow: 0 0 0 6px rgba(245,158,11,0.1); }
    100% { box-shadow: 0 0 0 2px rgba(245,158,11,0.2); }
}
.step-content { flex: 1; min-width: 0; }
.step-title {
    font-size: 14px; font-weight: 600; color: #1a1b22;
    margin-bottom: 2px;
}
.step-title.pending { color: #757684; font-weight: 400; }
.step-title.active { color: #d97706; }
.step-title.done { color: #15803d; }
.step-desc { font-size: 12px; color: #757684; line-height: 16px; }
.step-desc.active { color: #92400e; }

/* Terminal block */
.terminal-block {
    background: #1a1b22;
    border-radius: 8px;
    padding: 12px 14px;
    margin-top: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    line-height: 18px;
    color: #9ca3af;
    max-height: 120px;
    overflow-y: auto;
}
.terminal-line { margin: 0; }
.terminal-line.ok { color: #4ade80; }
.terminal-line.warn { color: #fbbf24; }
.terminal-line.active { color: #60a5fa; }

/* Stop button */
.stop-btn {
    background: transparent;
    border: 1px solid #fca5a5;
    border-radius: 8px;
    color: #ba1a1a;
    padding: 8px 16px;
    font-size: 13px; font-weight: 600;
    cursor: pointer;
    width: 100%;
    margin-top: 16px;
    display: flex; align-items: center; justify-content: center; gap: 6px;
    transition: all 0.15s;
}
.stop-btn:hover { background: #fef2f2; }

/* Stats row */
.stats-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 8px; margin-top: 16px;
}
.stat-card {
    background: #f4f2fc;
    border-radius: 8px;
    padding: 10px 12px;
    text-align: center;
}
.stat-num { font-family: 'Sora', sans-serif; font-size: 20px; font-weight: 700; color: #00288e; }
.stat-label { font-size: 11px; color: #757684; margin-top: 2px; }

/* ── Streamlit widget overrides ─────────────────────── */
div[data-testid="stTextArea"] textarea {
    font-family: 'Cairo', 'Inter', sans-serif !important;
    direction: rtl; font-size: 15px;
    border: 1px solid #e3e1eb !important;
    border-radius: 10px !important;
    background: #fbf8ff !important;
    resize: none !important;
    min-height: 80px !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #39b8fd !important;
    box-shadow: 0 0 0 2px rgba(57,184,253,0.15) !important;
}
div[data-testid="stTextArea"] label { display: none !important; }

.stButton > button {
    background: linear-gradient(135deg, #1e40af, #3755c3) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-family: 'Inter', sans-serif !important;
    font-size: 14px !important; font-weight: 600 !important;
    padding: 10px 20px !important; transition: opacity 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

[data-testid="stSidebar"] { display: none !important; }

div[data-testid="stSelectbox"] > div {
    border-radius: 8px !important;
    border-color: #e3e1eb !important;
    font-family: 'Inter', sans-serif !important;
}

.stExpander { border: 1px solid #e3e1eb !important; border-radius: 8px !important; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
defaults = {
    "history": [],
    "current_answer": "",
    "current_sources": [],
    "current_meta": {},
    "pipeline_state": "idle",   # idle | running | done | error
    "step_states": {},
    "terminal_lines": [],
    "stats": {"docs": 0, "chunks": 0, "time": 0},
    "prefill": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Pipeline steps definition ──────────────────────────────────────────────────
STEPS = [
    {"key": "analyze",   "title": "Receiving Question",       "desc": "Parsed intent and keywords.",         "icon": "🧠"},
    {"key": "scrape",    "title": "Searching Official Websites", "desc": "Identified relevant sources.",    "icon": "🌐"},
    {"key": "chunk",     "title": "Extracting Content",        "desc": "Cleaning HTML and Arabic text.",    "icon": "✂️"},
    {"key": "index",     "title": "Building Search Index",     "desc": "BM25 + Dense vector indexing.",     "icon": "📊"},
    {"key": "retrieve",  "title": "Retrieving Top Chunks",     "desc": "Hybrid similarity search.",         "icon": "🔎"},
    {"key": "generate",  "title": "Generating Answer",         "desc": "LLM synthesis with citations.",     "icon": "💡"},
]

def reset_pipeline():
    st.session_state.pipeline_state = "idle"
    st.session_state.step_states = {s["key"]: "pending" for s in STEPS}
    st.session_state.terminal_lines = []
    st.session_state.current_answer = ""
    st.session_state.current_sources = []
    st.session_state.stats = {"docs": 0, "chunks": 0, "time": 0}

reset_pipeline()

# ── Example questions ──────────────────────────────────────────────────────────
EXAMPLES = [
    ("🎒", "How can I enroll my child?",          "كيف يمكنني تسجيل طفلي في المدرسة؟"),
    ("📄", "School transfer documents?",           "ما هي وثائق نقل المدرسة؟"),
    ("📅", "Attendance policy?",                   "ما هي سياسة الحضور والغياب؟"),
    ("📚", "منهج رياضيات الصف الثالث الابتدائي؟", "منهج رياضيات الصف الثالث الابتدائي؟"),
    ("🎓", "جدول امتحانات الثانوية العامة؟",       "جدول امتحانات الثانوية العامة؟"),
]

# ── Render sidebar right (processing panel) ────────────────────────────────────
def render_processing_panel():
    state = st.session_state.pipeline_state
    live_class = "active" if state == "running" else ("idle" if state == "idle" else "")

    panel_html = f"""
    <div class="processing-header">
        <div class="processing-title">
            ⚙️ Live Processing
        </div>
        <div class="live-dot {live_class}"></div>
    </div>
    """
    st.markdown(panel_html, unsafe_allow_html=True)

    step_states = st.session_state.step_states
    terminal_lines = st.session_state.terminal_lines

    for i, step in enumerate(STEPS):
        s = step_states.get(step["key"], "pending")
        icon_map   = {"pending": "○", "active": "⟳", "done": "✓", "error": "✗"}
        icon       = icon_map.get(s, "○")
        icon_class = s
        title_class = s
        desc_class  = "active" if s == "active" else ""

        show_terminal = (s == "active" and step["key"] == "scrape" and terminal_lines)

        terminal_html = ""
        if show_terminal:
            lines_html = "".join(
                f'<p class="terminal-line {l.get("cls","")}">{l["text"]}</p>'
                for l in terminal_lines[-6:]
            )
            terminal_html = f'<div class="terminal-block">{lines_html}</div>'

        st.markdown(f"""
        <div class="step-row">
            <div class="step-icon-wrap {icon_class}">{icon}</div>
            <div class="step-content">
                <div class="step-title {title_class}">{step['title']}</div>
                <div class="step-desc {desc_class}">{step['desc']}</div>
                {terminal_html}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Stats
    if state == "done":
        s = st.session_state.stats
        st.markdown(f"""
        <div class="stats-row">
            <div class="stat-card">
                <div class="stat-num">{s['docs']}</div>
                <div class="stat-label">Documents</div>
            </div>
            <div class="stat-card">
                <div class="stat-num">{s['chunks']}</div>
                <div class="stat-label">Chunks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Build layout with columns ──────────────────────────────────────────────────
col_left, col_main, col_right = st.columns([1.1, 3.2, 1.4])

# ══════════════════════════════════════════════════════
# LEFT SIDEBAR
# ══════════════════════════════════════════════════════
with col_left:
    st.markdown("""
    <div class="sidebar-left">
        <div class="brand-logo">
            <div class="brand-icon">✦</div>
            <div class="brand-text">
                <div class="brand-name">EduSearch Egypt</div>
                <div class="brand-sub">AI Research Assistant</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if st.button("＋  New Research", key="new_research"):
        reset_pipeline()
        st.session_state.prefill = ""
        st.rerun()

    st.markdown("""
        <div class="nav-section-label">Navigation</div>
        <div class="nav-item active"><span class="nav-item-icon">🏠</span> Home</div>
        <div class="nav-section-label">History</div>
        <div class="nav-item"><span class="nav-item-icon">🕐</span> Recent Questions</div>
        <div class="nav-item"><span class="nav-item-icon">🔖</span> Saved Questions</div>
        <div class="nav-section-label">Resources</div>
        <div class="nav-item"><span class="nav-item-icon">📋</span> Official Sources</div>
        <div class="nav-item"><span class="nav-item-icon">📁</span> My Projects</div>
        <div class="sidebar-footer">
            <div class="nav-item"><span class="nav-item-icon">⚙️</span> Settings</div>
            <div class="nav-item"><span class="nav-item-icon">❓</span> Help Center</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# MAIN CENTER
# ══════════════════════════════════════════════════════
with col_main:
    st.markdown('<div class="main-center">', unsafe_allow_html=True)

    # Hero
    st.markdown("""
    <div class="hero-section">
        <div class="verified-badge">✓ VERIFIED OFFICIAL DATA</div>
        <h1 class="hero-title">Egypt Basic Education<br>AI Assistant</h1>
        <p class="hero-sub">Ask any question about curriculum, policies, or enrollment.<br>
        Get verified answers sourced from official Egyptian Ministry of Education documents.</p>
    </div>
    """, unsafe_allow_html=True)

    # Search box
    st.markdown('<div class="search-container">', unsafe_allow_html=True)

    prefill_val = st.session_state.pop("prefill", "") if "prefill" in st.session_state else ""
    question = st.text_area(
        "question",
        value=prefill_val,
        placeholder="e.g., What are the required documents for transferring a student to a secondary school in Cairo?",
        height=90,
        label_visibility="collapsed",
        key="question_input",
    )

    btn_col, strat_col = st.columns([2, 1])
    with btn_col:
        ask_clicked = st.button("Ask AI ▶", key="ask_btn")
    with strat_col:
        retriever_strategy = st.selectbox(
            "Strategy",
            options=["hybrid", "dense", "bm25"],
            format_func=lambda x: {"hybrid": "🔀 Hybrid", "dense": "🧠 Dense", "bm25": "🔤 BM25"}[x],
            label_visibility="collapsed",
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # Quick chips
    st.markdown('<div class="try-asking"><span class="try-label">Try asking:</span>', unsafe_allow_html=True)
    chip_cols = st.columns(len(EXAMPLES))
    for i, (icon, label, q) in enumerate(EXAMPLES):
        with chip_cols[i]:
            if st.button(f"{icon} {label}", key=f"chip_{i}"):
                st.session_state["prefill"] = q
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Run pipeline ───────────────────────────────────
    if ask_clicked and question.strip():
        reset_pipeline()
        st.session_state.pipeline_state = "running"
        t0 = time.time()

        # Placeholders for live updates
        meta_placeholder      = st.empty()
        answer_placeholder    = st.empty()
        sources_placeholder   = st.empty()

        # Step state updater
        def set_step(key: str, state: str, desc: str = ""):
            st.session_state.step_states[key] = state
            for s in STEPS:
                if s["key"] == key and desc:
                    s["desc"] = desc

        # Mark all pending
        for s in STEPS:
            st.session_state.step_states[s["key"]] = "pending"

        # ── Step 1: Analyze ──
        set_step("analyze", "active", "Parsing intent and keywords...")
        from backend.rag.orchestrator import get_pipeline_metadata
        meta = get_pipeline_metadata(question)
        st.session_state.current_meta = meta

        pills = "".join([
            f'<span class="meta-pill">{'🎓' if k=='grade' else '📚' if k=='subject' else '📅' if k=='term' else '🏷'} {v}</span>'
            for k, v in meta.items()
            if v and k not in ("search_query", "needs_live_search", "source_category")
        ])
        if pills:
            meta_placeholder.markdown(f'<div class="meta-row">{pills}</div>', unsafe_allow_html=True)

        set_step("analyze", "done", f"Subject: {meta.get('subject','—')} | Grade: {meta.get('grade','—')}")

        # ── Step 2: Scrape ──
        set_step("scrape", "active", "Fetching latest documents...")
        st.session_state.terminal_lines = [
            {"text": "> Connecting to ar.wikipedia.org...", "cls": "active"},
            {"text": "> Wikipedia API search...", "cls": "active"},
        ]

        from backend.scraper.query_analyzer import analyze_query
        from backend.scraper.live_scraper import scrape_for_query
        qmeta = analyze_query(question)
        raw_docs = scrape_for_query(qmeta)

        n_docs = len(raw_docs)
        st.session_state.stats["docs"] = n_docs
        st.session_state.terminal_lines = [
            {"text": f"> Scraped {n_docs} documents", "cls": "ok"},
            {"text": f"> Sources: {', '.join(set(d.metadata.get('source','wikipedia') for d in raw_docs))[:60]}", "cls": "ok"},
        ]
        if n_docs == 0:
            set_step("scrape", "error", "No documents found.")
            st.session_state.pipeline_state = "error"
            answer_placeholder.markdown(
                '<div class="answer-glass">⚠️ لم أتمكن من الوصول إلى المصادر. يرجى المحاولة مرة أخرى.</div>',
                unsafe_allow_html=True
            )
            st.stop()

        set_step("scrape", "done", f"Found {n_docs} document(s).")

        # ── Step 3: Chunk ──
        set_step("chunk", "active", "Cleaning HTML and Arabic text...")
        from backend.preprocessing.chunker import process_documents
        chunks = process_documents(raw_docs)
        n_chunks = len(chunks)
        st.session_state.stats["chunks"] = n_chunks
        if not chunks:
            set_step("chunk", "error", "No Arabic content extracted.")
            st.session_state.pipeline_state = "error"
            answer_placeholder.markdown(
                '<div class="answer-glass">⚠️ لم يتم استخراج محتوى كافٍ من المصادر.</div>',
                unsafe_allow_html=True
            )
            st.stop()
        set_step("chunk", "done", f"Extracted {n_chunks} chunks.")

        # ── Step 4: Index ──
        set_step("index", "active", "Building BM25 + FAISS index...")
        from backend.retrieval.retriever import build_retriever
        retriever = build_retriever(retriever_strategy)
        retriever.index(chunks)
        set_step("index", "done", f"Indexed {n_chunks} chunks ({retriever_strategy}).")

        # ── Step 5: Retrieve ──
        set_step("retrieve", "active", "Hybrid similarity search...")
        from config.settings import retrieval_cfg
        retrieved = retriever.search(question, retrieval_cfg.top_k_rerank)
        set_step("retrieve", "done", f"Retrieved {len(retrieved)} relevant chunks.")

        # ── Step 6: Generate ──
        set_step("generate", "active", "LLM synthesis with citations...")
        answer_placeholder.markdown(
            '<div class="answer-glass" id="answer-stream">⏳ جاري توليد الإجابة...</div>',
            unsafe_allow_html=True
        )

        from backend.rag.chain import stream_answer, generate_answer
        full_answer = ""
        try:
            for token in stream_answer(question, retrieved):
                full_answer += token
                answer_placeholder.markdown(
                    f'<div class="answer-glass" dir="rtl">{full_answer}▌</div>',
                    unsafe_allow_html=True
                )
        except Exception as exc:
            full_answer = f"⚠️ خطأ: {exc}"
            logger.error(f"Stream error: {exc}")

        # Final answer (no cursor)
        answer_placeholder.markdown(
            f'<div class="answer-glass" dir="rtl">{full_answer}</div>',
            unsafe_allow_html=True
        )

        set_step("generate", "done", "Answer generated successfully.")
        st.session_state.pipeline_state = "done"
        st.session_state.current_answer = full_answer
        st.session_state.stats["time"] = round(time.time() - t0, 1)

        # Sources
        sources = [
            {"url": rc.chunk.source_url, "title": rc.chunk.title or rc.chunk.source_url,
             "score": round(rc.score, 4)}
            for rc in retrieved
        ]
        st.session_state.current_sources = sources
        if sources:
            badges = "".join([
                f'<a class="source-chip" href="{s["url"]}" target="_blank">🔗 {s["title"][:45]}</a>'
                for s in sources
            ])
            sources_placeholder.markdown(
                f'<div class="sources-row">{badges}</div>', unsafe_allow_html=True
            )

        # Save to history
        st.session_state.history.insert(0, {
            "question": question,
            "answer": full_answer,
            "sources": sources,
            "meta": meta,
        })

    elif ask_clicked and not question.strip():
        st.warning("⚠️ Please type your question first.")

    # ── Show previous answer if exists ────────────────
    elif st.session_state.current_answer and st.session_state.pipeline_state == "done":
        meta = st.session_state.current_meta
        pills = "".join([
            f'<span class="meta-pill">{v}</span>'
            for k, v in meta.items()
            if v and k not in ("search_query", "needs_live_search", "source_category")
        ])
        if pills:
            st.markdown(f'<div class="meta-row">{pills}</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="answer-glass" dir="rtl">{st.session_state.current_answer}</div>',
            unsafe_allow_html=True
        )
        if st.session_state.current_sources:
            badges = "".join([
                f'<a class="source-chip" href="{s["url"]}" target="_blank">🔗 {s["title"][:45]}</a>'
                for s in st.session_state.current_sources
            ])
            st.markdown(f'<div class="sources-row">{badges}</div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="empty-state">
            Start by asking a question to search official databases.
        </div>
        """, unsafe_allow_html=True)

    # ── History ────────────────────────────────────────
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.markdown("### 🕘 Recent Questions")
        for item in st.session_state.history[1:4]:
            q = item["question"]
            a = item["answer"]
            with st.expander(q[:80] + ("…" if len(q) > 80 else "")):
                st.markdown(f'<div class="answer-glass" dir="rtl">{a}</div>', unsafe_allow_html=True)
                if item.get("sources"):
                    badges = "".join([
                        f'<a class="source-chip" href="{s["url"]}" target="_blank">{s["title"][:40]}</a>'
                        for s in item["sources"]
                    ])
                    st.markdown(f'<div class="sources-row">{badges}</div>', unsafe_allow_html=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            reset_pipeline()
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# RIGHT SIDEBAR — Live Processing Panel
# ══════════════════════════════════════════════════════
with col_right:
    st.markdown('<div class="sidebar-right">', unsafe_allow_html=True)
    render_processing_panel()

    if st.session_state.pipeline_state == "running":
        if st.button("⬛ Stop Processing", key="stop_btn"):
            st.session_state.pipeline_state = "idle"
            st.rerun()

    if st.session_state.pipeline_state == "done":
        t = st.session_state.stats.get("time", 0)
        st.markdown(f"""
        <div style="margin-top:16px;padding:12px;background:#f0fdf4;
             border:1px solid #bbf7d0;border-radius:8px;text-align:center">
            <div style="font-size:13px;color:#15803d;font-weight:600">
                ✓ Completed in {t}s
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
