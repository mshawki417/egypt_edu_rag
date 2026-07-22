"""
Egypt Education RAG
Real-Time Streamlit Frontend

Author:
Mustafa Shawki
"""


from __future__ import annotations


import sys

from pathlib import Path


import streamlit as st



ROOT_DIR = Path(__file__).resolve().parent.parent


sys.path.insert(
    0,
    str(ROOT_DIR)
)



from backend.rag.orchestrator import run_rag_pipeline





# ==========================
# Config
# ==========================


st.set_page_config(

    page_title="Egypt Education AI",

    layout="wide",

    initial_sidebar_state="expanded"

)






# ==========================
# CSS
# ==========================


def load_css():


    css_file = (

        ROOT_DIR /
        "frontend" /
        "assets" /
        "style.css"

    )


    if css_file.exists():


        st.markdown(

            f"""

            <style>

            {css_file.read_text(
                encoding="utf-8"
            )}

            </style>

            """,

            unsafe_allow_html=True

        )



load_css()






# ==========================
# Session
# ==========================


if "messages" not in st.session_state:

    st.session_state.messages=[]





if "pipeline_cache" not in st.session_state:

    st.session_state.pipeline_cache={}







# ==========================
# Sidebar
# ==========================


with st.sidebar:



    st.markdown(

        """

        <div class="sidebar-title">

        Egypt Education AI

        </div>

        """,

        unsafe_allow_html=True

    )



    st.write(

        """

        نظام RAG متخصص في التعليم المصري.

        يبحث في المصادر ويولد إجابات
        مدعومة بالمراجع.

        """

    )


    st.divider()



    if st.button(

        "Clear Chat",

        use_container_width=True

    ):


        st.session_state.messages=[]

        st.session_state.pipeline_cache={}

        st.rerun()






# ==========================
# Header
# ==========================


st.markdown(

"""

<div class="main-header">


<h1>
Egypt Education AI Assistant
</h1>


<p>
Real-Time Retrieval Augmented Generation System
</p>


</div>

""",

unsafe_allow_html=True

)






# ==========================
# History
# ==========================


for msg in st.session_state.messages:


    with st.chat_message(
        msg["role"]
    ):


        st.markdown(
            msg["content"]
        )


        if msg.get("sources"):


            with st.expander(
                "Sources"
            ):


                for src in msg["sources"]:

                    st.write(src)







# ==========================
# Input
# ==========================


query = st.chat_input(

    "اكتب سؤالك..."

)




if query:



    st.session_state.messages.append(

        {

        "role":"user",

        "content":query

        }

    )



    with st.chat_message("user"):

        st.markdown(query)




    with st.chat_message(
        "assistant"
    ):



        status_box = st.empty()



        answer_box = st.empty()



        full_answer=""



        def update_status(step):


            mapping={


                "step:analyze":
                "Analyzing question...",


                "step:scrape":
                "Searching live sources...",


                "step:chunk":
                "Processing documents...",


                "step:retrieve":
                "Finding relevant context...",


                "step:generate":
                "Generating answer..."

            }



            status_box.info(

                mapping.get(
                    step,
                    step
                )

            )





        result = run_rag_pipeline(

            query,

            stream=True,

            status_callback=update_status

        )



        for token in result:


            full_answer += token


            answer_box.markdown(
                full_answer
            )



        status_box.empty()





    st.session_state.messages.append(

        {

        "role":"assistant",

        "content":full_answer,

        "sources":[]

        }

    )
