from retrieval.hybrid_retriever import retrieve_faiss_candidates
from retrieval.bm25_retriever import retrieve_bm25
from retrieval.rrf import reciprocal_rank_fusion

q = "¿Para qué sirve el paracetamol?"

faiss = retrieve_faiss_candidates(q, top_k=10)
bm25 = retrieve_bm25(q, top_k=10)
rrf = reciprocal_rank_fusion([faiss, bm25], top_k=10)

print("\nFAISS")
for c in faiss:
    print(c["faiss_rank"], c["chunk_id"], c["score"], c["text"][:80])

print("\nBM25")
for c in bm25:
    print(c["bm25_rank"], c["chunk_id"], c["bm25_score"], c["text"][:80])

print("\nRRF")
for c in rrf:
    print(c["rrf_rank"], c["chunk_id"], c["rrf_score"], c["rrf_sources"], c["text"][:80])