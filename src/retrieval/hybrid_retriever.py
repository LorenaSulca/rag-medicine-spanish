import faiss

from retrieval.retrieval_faiss import (
    embed_query,
    build_index_variant,
    load_faiss,
    extract_query_signals,
    filter_by_medical_signals,
    TOP_K,
)

from .medspaner_bridge import run_medspaner_question
from .bm25_retriever import retrieve_bm25
from .rrf import reciprocal_rank_fusion


def retrieve_faiss_candidates(
    query_text: str,
    top_k: int = 10,
    chunking_variant: str = "sections",
    embedding_model: str = "openai",
) -> list[dict]:
    """
    Recupera candidatos usando FAISS para la variante experimental indicada.
    """

    index_variant = build_index_variant(
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
    )

    query_emb = embed_query(
        text=query_text,
        embedding_model=embedding_model,
    )

    query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)

    index, metadata = load_faiss(index_variant=index_variant)

    scores, idxs = index.search(query_emb, top_k)

    candidates = []

    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        meta = dict(metadata[idx])
        meta["score"] = float(score)
        meta["faiss_score"] = float(score)
        meta["faiss_rank"] = rank
        meta["index_variant"] = meta.get("index_variant", index_variant)
        meta["chunking_strategy"] = meta.get("chunking_strategy", chunking_variant)
        meta["embedding_model"] = meta.get("embedding_model", embedding_model)
        meta["_retrieval_source"] = "faiss"

        candidates.append(meta)

    return candidates


def retrieve_hybrid(
    query_text: str,
    top_k: int = TOP_K,
    candidate_k: int = 10,
    dynamic_k: bool = False,
    chunking_variant: str = "sections",
    embedding_model: str = "openai",
):
    """
    Retrieval híbrido:
    1. MEDSPANER sobre la consulta.
    2. Recuperación semántica con FAISS.
    3. Recuperación léxica con BM25.
    4. Fusión con RRF.
    5. Reranking heurístico por señales biomédicas.
    """

    medspaner_output = run_medspaner_question(query_text)
    signals = extract_query_signals(medspaner_output)

    final_k = top_k

    faiss_candidates = retrieve_faiss_candidates(
        query_text=query_text,
        top_k=candidate_k,
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
    )

    bm25_candidates = retrieve_bm25(
        query=query_text,
        top_k=candidate_k,
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
    )

    fused_candidates = reciprocal_rank_fusion(
        ranked_lists=[faiss_candidates, bm25_candidates],
        rrf_k=60,
        top_k=candidate_k,
    )

    refined = filter_by_medical_signals(fused_candidates, signals)

    return refined[:final_k], signals, medspaner_output