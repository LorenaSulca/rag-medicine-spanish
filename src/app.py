import streamlit as st
from openai import OpenAI

from rag import default_rag_client
from ui.sidebar import render_sidebar
from ui.components import render_response


st.set_page_config(
    page_title="RAG Médico Demo",
    layout="wide",
)

st.title("RAG Médico - Panel de Prueba")

question, experiment, run = render_sidebar()

if run and question:
    client = OpenAI()

    rag = default_rag_client(
        client,
        experiment=experiment,
    )

    with st.spinner("Procesando..."):
        response = rag.query(question)

    render_response(response)