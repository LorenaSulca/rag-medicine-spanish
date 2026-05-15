import streamlit as st
from ui.formatters import (
    format_validation_summary,
    format_chunk_title,
)


def render_response(response: dict):
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Respuesta")
        st.write(response.get("answer"))

        st.subheader("Estado")
        st.write(response.get("status"))

        st.subheader("Chunks")

        for chunk in response.get("chunks", []):
            title = format_chunk_title(chunk)

            with st.expander(title):
                st.write(chunk.get("text"))

                st.json({
                    "score": chunk.get("score"),
                    "rrf_rank": chunk.get("rrf_rank"),
                    "rrf_sources": chunk.get("rrf_sources"),
                    "dynamic_k": chunk.get("dynamic_k"),
                    "query_complexity": chunk.get("query_complexity"),
                })

    with col2:
        st.subheader("Validación")

        validation = response.get("validation")

        if not validation:
            st.info("Sin validación")
            return

        st.markdown(format_validation_summary(validation))

        st.subheader("Detalle")
        st.json(validation)