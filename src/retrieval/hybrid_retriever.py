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
from .noise import inject_noise_chunks


def retrieve_faiss_candidates(
    query_text: str,
    top_k: int = 10,
    chunking_variant: str = "sections",
    embedding_model: str = "openai",
    index_suffix: str | None = None,
) -> list[dict]:
    index_variant = build_index_variant(
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
        index_suffix=index_suffix,
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
        if idx < 0 or idx >= len(metadata):
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
    index_suffix: str | None = None,
    noise_injection: bool = False,
    noise_chunks: int = 2,
    noise_seed: int = 42,
    noise_placement: str = "end",
):
    """
    Retrieval híbrido:
    1. MEDSPANER sobre la consulta.
    2. FAISS.
    3. BM25.
    4. RRF.
    5. Reranking biomédico.
    6. Inyección opcional de ruido semántico.
    """

    medspaner_output = run_medspaner_question(query_text)
    signals = extract_query_signals(medspaner_output)

    final_k = top_k

    index_variant = build_index_variant(
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
        index_suffix=index_suffix,
    )

    faiss_candidates = retrieve_faiss_candidates(
        query_text=query_text,
        top_k=candidate_k,
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
        index_suffix=index_suffix,
    )

    bm25_candidates = retrieve_bm25(
        query=query_text,
        top_k=candidate_k,
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
        index_suffix=index_suffix,
    )

    fused_candidates = reciprocal_rank_fusion(
        ranked_lists=[faiss_candidates, bm25_candidates],
        rrf_k=60,
        top_k=candidate_k,
    )

    refined = filter_by_medical_signals(fused_candidates, signals)
    clean_context = refined[:final_k]

    if not noise_injection:
        return clean_context, signals, medspaner_output

    _, metadata = load_faiss(index_variant=index_variant)

    noisy_context = inject_noise_chunks(
        retrieved_chunks=clean_context,
        metadata=metadata,
        noise_chunks=noise_chunks,
        seed=noise_seed,
        placement=noise_placement,
    )

    for ch in noisy_context:
        ch["_noise_injection_enabled"] = True
        ch["_noise_chunks_requested"] = noise_chunks
        ch["_noise_placement"] = noise_placement

    return noisy_context, signals, medspaner_output