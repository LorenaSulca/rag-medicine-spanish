def get_chunk_key(chunk: dict) -> str:
    """
    Genera una clave estable para identificar un chunk entre rankings.
    """
    document_id = chunk.get("document_id", "desconocido")
    chunk_id = chunk.get("chunk_id", "?")
    return f"{document_id}::{chunk_id}"


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    rrf_k: int = 60,
    top_k: int = 5,
) -> list[dict]:
    """
    Fusiona múltiples rankings usando Reciprocal Rank Fusion.

    score_RRF(d) = sum(1 / (rrf_k + rank_i))

    donde rank_i es la posición del documento en cada ranking.
    """

    fused = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = get_chunk_key(chunk)

            if key not in fused:
                fused[key] = {
                    "chunk": dict(chunk),
                    "rrf_score": 0.0,
                    "sources": [],
                }

            fused[key]["rrf_score"] += 1.0 / (rrf_k + rank)
            fused[key]["sources"].append({
                "rank": rank,
                "source": chunk.get("_retrieval_source", "unknown"),
            })

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    results = []

    for rank, item in enumerate(ranked[:top_k], start=1):
        chunk = item["chunk"]
        chunk["rrf_score"] = item["rrf_score"]
        chunk["rrf_rank"] = rank
        chunk["rrf_sources"] = item["sources"]

        # Score estándar para mantener compatibilidad con el reranking biomédico existente
        chunk["score"] = float(item["rrf_score"])
        results.append(chunk)

    return results