"""
Egypt Education RAG
Production Streamlit Frontend

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



from backend.rag.orchestrator import (
    run_rag_pipeline,
    clear_pipeline_cache
)





# ==========================
# Config
# ==========================

st.set_page_config(

    page_title="Egypt Education AI",

    page_icon=None,

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
# Session State
# ==========================

if "messages" not in st.session_state:

    st.session_state.messages = []



if "processing" not in st.session_state:

    st.session_state.processing = False






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

        يسترجع المعلومات من المصادر
        ويولد إجابات مدعومة بالمراجع.

        """

    )


    st.divider()



    if st.button(

        "Clear Chat",

        use_container_width=True

    ):


        st.session_state.messages = []


        try:

            clear_pipeline_cache()

        except Exception:

            pass


        st.rerun()






    if st.button(

        "Reset System",

        use_container_width=True

    ):


        st.session_state.clear()


        try:

            clear_pipeline_cache()

        except Exception:

            pass


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
Real-Time Retrieval Augmented Generation
</p>


</div>

""",

unsafe_allow_html=True

)







# ==========================
# Chat History
# ==========================

for message in st.session_state.messages:


    with st.chat_message(

        message["role"]

    ):


        st.markdown(

            message["content"]

        )


        if message.get("sources"):


            with st.expander(

                "Sources"

            ):


                for source in message["sources"]:


                    st.write(

                        source

                    )








# ==========================
# User Input
# ==========================

query = st.chat_input(

    "اكتب سؤالك..."

)



if query and not st.session_state.processing:


    st.session_state.processing = True



    st.session_state.messages.append(

        {

            "role":

            "user",

            "content":

            query

        }

    )



    with st.chat_message(

        "user"

    ):


        st.markdown(

            query

        )






    with st.chat_message(

        "assistant"

    ):


        status = st.empty()


        answer_area = st.empty()



        def update_status(step):

            status.info(step)





        try:


            result = run_rag_pipeline(

                query,

                stream=False,

                status_callback=update_status

            )



            status.empty()



            answer = result.answer



            answer_area.markdown(

                answer

            )





            if result.sources:


                with st.expander(

                    "Sources"

                ):


                    for src in result.sources:


                        st.write(

                            src

                        )





            st.session_state.messages.append(

                {

                    "role":

                    "assistant",


                    "content":

                    answer,


                    "sources":

                    result.sources

                }

            )




        except Exception as e:


            logger_message = (

                f"Pipeline Error: {str(e)}"

            )


            status.error(

                logger_message

            )


            st.session_state.messages.append(

                {

                    "role":

                    "assistant",

                    "content":

                    logger_message,

                    "sources":[]

                }

            )



        finally:


            st.session_state.processing = False
