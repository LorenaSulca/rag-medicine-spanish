import numpy as np
import faiss

from retrieval.retrieval_faiss import (
    embed,
    load_faiss,
    extract_query_signals,
    filter_by_medical_signals,
    TOP_K,
)

from .medspaner_bridge import run_medspaner_question
from .bm25_retriever import retrieve_bm25
from .rrf import reciprocal_rank_fusion


def retrieve_faiss_candidates(query_text: str, top_k: int = 10) -> list[dict]:
    """
    Recupera candidatos usando FAISS, equivalente al retrieval semántico actual,
    pero permite obtener más de TOP_K para luego fusionar con BM25.
    """

    query_emb = embed(query_text)
    query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)

    index, metadata, _ = load_faiss()

    scores, idxs = index.search(query_emb, top_k)

    candidates = []

    for rank, (score, idx) in enumerate(zip(scores[0], idxs[0]), start=1):
        if idx < 0:
            continue

        meta = dict(metadata[idx])
        meta["score"] = float(score)
        meta["faiss_score"] = float(score)
        meta["faiss_rank"] = rank
        meta["_retrieval_source"] = "faiss"
        candidates.append(meta)

    return candidates


def retrieve_hybrid(
    query_text: str,
    top_k: int = TOP_K,
    candidate_k: int = 10,
    dynamic_k: bool = False,
):
    """
    Retrieval híbrido:
    1. MEDSPANER sobre la consulta.
    2. Recuperación semántica con FAISS.
    3. Recuperación léxica con BM25.
    4. Fusión con RRF.
    5. Reranking heurístico por señales biomédicas.
    """

    # 1. Análisis biomédico de la pregunta
    medspaner_output = run_medspaner_question(query_text)
    signals = extract_query_signals(medspaner_output)

    # De momento dynamic_k no se usa en Propuesta 1.
    # Se deja el parámetro para compatibilidad futura.
    final_k = top_k

    # 2. Recuperación semántica
    faiss_candidates = retrieve_faiss_candidates(
        query_text=query_text,
        top_k=candidate_k,
    )

    # 3. Recuperación léxica
    bm25_candidates = retrieve_bm25(
        query=query_text,
        top_k=candidate_k,
    )

    # 4. Fusión RRF
    fused_candidates = reciprocal_rank_fusion(
        ranked_lists=[faiss_candidates, bm25_candidates],
        rrf_k=60,
        top_k=candidate_k,
    )

    # 5. Reranking biomédico existente
    refined = filter_by_medical_signals(fused_candidates, signals)

    return refined[:final_k], signals, medspaner_output