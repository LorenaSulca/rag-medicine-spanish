from .retrieval_faiss import retrieve_chunks


def retrieve_base(
    question: str,
    index_variant: str = "sections",
):
    """
    Wrapper del retrieval base.
    No cambia la lógica del baseline; solo permite elegir
    la variante de índice experimental: flat o sections.
    """

    chunks, signals, medspaner_raw = retrieve_chunks(
        query_text=question,
        index_variant=index_variant,
    )

    return chunks, signals, medspaner_raw