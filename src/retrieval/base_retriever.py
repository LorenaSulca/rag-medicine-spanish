from .retrieval_faiss import retrieve_chunks


def retrieve_base(question: str):
    """
    Wrapper del retrieval actual.
    No cambia comportamiento, solo adapta el formato.
    """

    chunks, signals, medspaner_raw = retrieve_chunks(question)

    # chunks ya viene como lista de dicts con text, score, etc.
    return chunks, signals, medspaner_raw