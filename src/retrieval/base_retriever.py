from .retrieval_faiss import retrieve_chunks


def retrieve_base(
    question: str,
    chunking_variant: str = "sections",
    embedding_model: str = "openai",
):
    """
    Wrapper del retrieval base.

    Permite elegir:
    - chunking_variant: flat / sections
    - embedding_model: openai / multilingual_e5 / medcpt
    """

    chunks, signals, medspaner_raw = retrieve_chunks(
        query_text=question,
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
    )

    return chunks, signals, medspaner_raw