import re
from rank_bm25 import BM25Okapi

from retrieval.retrieval_faiss import load_faiss


def tokenize(text: str) -> list[str]:
    """
    Tokenizador simple para BM25.
    Convierte a minúsculas y conserva tokens alfanuméricos.
    """
    if not text:
        return []

    return re.findall(r"\b\w+\b", text.lower())


def retrieve_bm25(query: str, top_k: int = 10) -> list[dict]:
    """
    Recuperación léxica usando BM25 sobre los chunks de metadata.json.
    """

    _, metadata, _ = load_faiss()

    corpus_texts = [item.get("text", "") for item in metadata]
    tokenized_corpus = [tokenize(text) for text in corpus_texts]

    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    results = []

    for rank, idx in enumerate(ranked_indices, start=1):
        item = dict(metadata[idx])
        item["bm25_score"] = float(scores[idx])
        item["bm25_rank"] = rank
        item["_retrieval_source"] = "bm25"
        results.append(item)

    return results