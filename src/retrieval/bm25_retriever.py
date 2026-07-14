import re

from rank_bm25 import BM25Okapi

from retrieval.retrieval_faiss import (
    build_index_variant,
    load_faiss,
)


def tokenize(text: str) -> list[str]:
    """
    Tokenizador simple para BM25.
    Convierte a minúsculas y conserva tokens alfanuméricos.
    """

    if not text:
        return []

    return re.findall(r"\b\w+\b", text.lower())


def retrieve_bm25(
    query: str,
    top_k: int = 10,
    chunking_variant: str = "sections",
    embedding_model: str = "openai",
    index_suffix: str | None = None,
) -> list[dict]:
    """
    Recuperación léxica BM25 sobre metadata.json correspondiente
    a la variante experimental seleccionada.

    Nota:
    BM25 no usa embeddings directamente, pero sí debe leer la metadata
    del mismo índice experimental que FAISS para mantener la comparación limpia.
    """

    index_variant = build_index_variant(
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
        index_suffix=index_suffix,
    )

    _, metadata = load_faiss(index_variant=index_variant)

    corpus_texts = [
        item.get("text", "")
        for item in metadata
    ]

    tokenized_corpus = [
        tokenize(text)
        for text in corpus_texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )[:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        item = dict(metadata[idx])

        item["bm25_score"] = float(scores[idx])
        item["bm25_rank"] = rank
        item["index_variant"] = item.get("index_variant", index_variant)
        item["chunking_strategy"] = item.get("chunking_strategy", chunking_variant)
        item["embedding_model"] = item.get("embedding_model", embedding_model)
        item["_retrieval_source"] = "bm25"

        results.append(item)

    return results