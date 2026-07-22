"""
Egypt Education RAG
Professional Streamlit Frontend

Author: Mustafa Shawki
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path
import sys

# Add project root
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

from backend.rag_pipeline import RAGPipeline


# =========================
# Page Configuration
# =========================

st.set_page_config(
    page_title="Egypt Education AI Assistant",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================
# Load CSS
# =========================

def load_css():

    css_path = ROOT_DIR / "assets" / "style.css"

    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )


load_css()


# =========================
# Initialize RAG
# =========================

@st.cache_resource
def load_pipeline():

    return RAGPipeline()


pipeline = load_pipeline()


# =========================
# Session State
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================
# Sidebar
# =========================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-title">
        Egypt Education AI
        </div>
        """,
        unsafe_allow_html=True
    )


    st.markdown(
        """
        نظام ذكاء اصطناعي للإجابة
        على الأسئلة المتعلقة بالتعليم المصري
        باستخدام تقنية RAG.
        """
    )


    st.divider()


    if st.button(
        "Clear Conversation",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.rerun()


    st.divider()


    st.caption(
        "Retrieval Augmented Generation System"
    )



# =========================
# Header
# =========================

st.markdown(
    """
    <div class="main-header">

        <h1>
        Egypt Education AI Assistant
        </h1>

        <p>
        Ask questions about Egyptian education policies,
        universities, regulations and documents.
        </p>

    </div>
    """,
    unsafe_allow_html=True
)



# =========================
# Chat History
# =========================


for message in st.session_state.messages:

    role = message["role"]

    with st.chat_message(role):

        st.markdown(
            message["content"]
        )


        if "sources" in message:

            with st.expander(
                "Retrieved Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        source
                    )



# =========================
# User Input
# =========================


query = st.chat_input(
    "اكتب سؤالك هنا..."
)


if query:


    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )


    with st.chat_message("user"):

        st.markdown(query)



    with st.chat_message("assistant"):


        with st.spinner(
            "Searching knowledge base..."
        ):


            response = pipeline.run(
                query
            )


        answer = response.get(
            "answer",
            "No answer found."
        )


        sources = response.get(
            "sources",
            []
        )


        st.markdown(answer)



        if sources:

            with st.expander(
                "Retrieved Sources"
            ):

                for src in sources:

                    st.write(src)



    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )
