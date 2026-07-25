"""
Egypt Education RAG
Production Streamlit Frontend — ChatGPT Style (Arabic RTL)

Author: Mustafa Shawki
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

# ──────────────────────────────────────────────────────────
# Path Setup
# ──────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

try:
    from backend.rag.orchestrator import run_rag_pipeline, clear_pipeline_cache
    BACKEND_AVAILABLE = True
except ImportError:
    BACKEND_AVAILABLE = False

# ──────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="نظام التعليم المصري AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# CSS Injection
# ──────────────────────────────────────────────────────────
_css_file = ROOT_DIR / "frontend" / "assets" / "style.css"
if _css_file.exists():
    st.markdown(
        f"<style>{_css_file.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────
# Session State Init
# ──────────────────────────────────────────────────────────
def _init_state() -> None:
    defaults = {
        "messages": [],
        "conversations": [
            {"id": 1, "title": "استفسار عن المناهج الدراسية", "active": True},
            {"id": 2, "title": "قواعد القبول في الجامعات", "active": False},
            {"id": 3, "title": "نظام الثانوية العامة", "active": False},
        ],
        "processing": False,
        "current_conv_id": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()

# ──────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────
SUGGESTIONS = [
    ("📚", "ما هي مواد الثانوية العامة؟"),
    ("🎓", "كيف يعمل نظام التنسيق؟"),
    ("📝", "ما هي اشتراطات القبول في الطب؟"),
    ("🏫", "ما الفرق بين التعليم الأزهري والحكومي؟"),
]

def _render_welcome() -> None:
    """Show welcome screen when no messages exist."""
    st.markdown(
        """
        <div class="welcome-container">
            <div class="welcome-icon">🎓</div>
            <div class="welcome-title">نظام التعليم المصري AI</div>
            <div class="welcome-subtitle">
                مساعدك الذكي المتخصص في التعليم المصري.<br>
                اسأل عن المناهج، التنسيق، القبول، أو أي استفسار تعليمي.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Suggestion chips as real buttons
    cols = st.columns(2)
    for i, (icon, label) in enumerate(SUGGESTIONS):
        with cols[i % 2]:
            if st.button(f"{icon}  {label}", key=f"sug_{i}", use_container_width=True):
                _handle_query(label)
                st.rerun()


def _confidence_class(score: float) -> str:
    return "med" if score < 85 else ""


def _render_source_cards(sources: list) -> None:
    """Render source cards as custom HTML."""
    if not sources:
        return
    cards_html = '<div class="sources-container">'
    cards_html += '<div class="sources-title">📎 المصادر المستخدمة</div>'
    for src in sources:
        # src may be a string or dict
        if isinstance(src, dict):
            name      = src.get("name", "مصدر")
            ref       = src.get("ref", "")
            snippet   = src.get("snippet", str(src))
            confidence = src.get("confidence", 90)
            icon      = "📄" if "pdf" in name.lower() else "📊"
        else:
            name, ref, snippet, confidence, icon = str(src), "", "", 90, "📄"

        cls = _confidence_class(confidence)
        cards_html += f"""
        <div class="source-card">
            <div class="source-card-header">
                <div class="source-name">
                    <span class="source-icon">{icon}</span>
                    {name}
                </div>
                <span class="confidence-badge {cls}">{confidence}%</span>
            </div>
            {"" if not ref else f'<div class="source-ref">{ref}</div>'}
            {"" if not snippet else f'<div class="source-snippet">{snippet}</div>'}
        </div>
        """
    cards_html += "</div>"
    st.markdown(cards_html, unsafe_allow_html=True)


def _handle_query(query: str) -> None:
    """Process a user query through the RAG pipeline."""
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.processing = True

    if not BACKEND_AVAILABLE:
        # Demo mode fallback
        time.sleep(1.2)
        demo_answer = (
            f"**إجابة تجريبية على سؤالك:** {query}\n\n"
            "هذا وضع العرض التوضيحي — الـ backend غير متاح حالياً. "
            "قم بتوصيل `backend.rag.orchestrator` للحصول على إجابات حقيقية."
        )
        st.session_state.messages.append({
            "role": "assistant",
            "content": demo_answer,
            "sources": [],
        })
    else:
        steps_log = []

        def _status_cb(step: str) -> None:
            steps_log.append(step)

        try:
            result = run_rag_pipeline(
                query,
                stream=False,
                status_callback=_status_cb,
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": result.answer,
                "sources": result.sources or [],
            })
        except Exception as exc:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"⚠️ خطأ في معالجة طلبك: {exc}",
                "sources": [],
            })

    st.session_state.processing = False

# ──────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────
with st.sidebar:
    # Logo / Header
    st.markdown(
        """
        <div class="sidebar-title">نظام التعليم المصري</div>
        <div style="font-size:11px;color:#6b7280;padding:0 16px 14px;
                    border-bottom:1px solid rgba(255,255,255,0.07);">
            RAG — استرجاع معزز بالمراجع
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # New conversation button
    if st.button("✏️  محادثة جديدة", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.session_state.current_conv_id = None
        st.rerun()

    st.divider()

    # Conversations list
    st.markdown(
        '<div class="sidebar-section-label">المحادثات الأخيرة</div>',
        unsafe_allow_html=True,
    )

    for conv in st.session_state.conversations:
        is_active = conv["id"] == st.session_state.get("current_conv_id")
        active_cls = "active" if is_active else ""
        st.markdown(
            f"""
            <div class="sidebar-conv-item {active_cls}">
                <span class="sidebar-conv-icon">💬</span>
                {conv["title"]}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # System info
    st.markdown(
        """
        <div style="padding:0 4px">
          <p style="font-size:12px;color:#6b7280;line-height:1.7;direction:rtl;text-align:right;">
            يسترجع المعلومات من المصادر الموثوقة
            ويولد إجابات مدعومة بالمراجع التعليمية المصرية.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='flex:1'></div>", unsafe_allow_html=True)

    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ مسح", use_container_width=True):
            st.session_state.messages = []
            try:
                if BACKEND_AVAILABLE:
                    clear_pipeline_cache()
            except Exception:
                pass
            st.rerun()
    with col2:
        if st.button("🔄 إعادة", use_container_width=True):
            st.session_state.clear()
            try:
                if BACKEND_AVAILABLE:
                    clear_pipeline_cache()
            except Exception:
                pass
            st.rerun()

# ──────────────────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────────────────

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🎓 نظام التعليم المصري AI</h1>
        <p>استرجاع معزز بالمراجع — Real-Time RAG</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Chat history ──────────────────────────────────────────
if not st.session_state.messages:
    _render_welcome()
else:
    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]
        sources = msg.get("sources", [])

        with st.chat_message(role, avatar="🎓" if role == "assistant" else "👤"):
            st.markdown(content)
            if sources and role == "assistant":
                with st.expander("📎 عرض المصادر"):
                    _render_source_cards(sources)

# ── Typing indicator while processing ─────────────────────
if st.session_state.processing:
    st.markdown(
        """
        <div class="typing-indicator">
            <div style="width:34px;height:34px;background:linear-gradient(135deg,#4f7cff,#1d4ed8);
                        border-radius:8px;display:flex;align-items:center;
                        justify-content:center;font-size:16px;flex-shrink:0;">🎓</div>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Chat Input ────────────────────────────────────────────
query = st.chat_input("اكتب سؤالك هنا... (مثال: ما هي مواد الثانوية العامة؟)")

if query and not st.session_state.processing:
    _handle_query(query)
    st.rerun()

# Disclaimer
st.markdown(
    '<div class="chat-disclaimer">'
    'قد يُخطئ الذكاء الاصطناعي — تحقق دائماً من المصادر الرسمية'
    '</div>',
    unsafe_allow_html=True,
)
