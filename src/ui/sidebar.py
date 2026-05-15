import streamlit as st


EXPERIMENTS = [
    "baseline",
    "p1_retrieval",
    "p1_citations",
    "propuesta_1_full",
    "p2_dynamic_retrieval",
    "p2_refine",
    "propuesta_2_full",
]


def render_sidebar():
    with st.sidebar:
        st.header("Configuración")

        experiment = st.selectbox(
            "Selecciona configuración",
            EXPERIMENTS,
        )

        question = st.text_area(
            "Pregunta",
            placeholder="Ej: ¿Cuál es la dosis de paracetamol?",
            height=120,
        )

        run = st.button("Consultar")

    return question, experiment, run