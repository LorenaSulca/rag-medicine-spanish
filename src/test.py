from openai import OpenAI
from rag import default_rag_client

client = OpenAI()

rag = default_rag_client(
    client,
    experiment="p1_retrieval"
)

response = rag.query("¿Para qué sirve el paracetamol?")

print("\n=== ANSWER ===\n")
print(response["answer"])

print("\n=== STATUS ===\n")
print(response["status"])

print("\n=== CHUNKS ===\n")
for ch in response["chunks"]:
    print({
        "document_id": ch.get("document_id"),
        "chunk_id": ch.get("chunk_id"),
        "score": ch.get("score"),
        "faiss_rank": ch.get("faiss_rank"),
        "bm25_rank": ch.get("bm25_rank"),
        "rrf_score": ch.get("rrf_score"),
        "rrf_rank": ch.get("rrf_rank"),
        "rrf_sources": ch.get("rrf_sources"),
        "_rerank_score": ch.get("_rerank_score"),
    })

print("\n=== ERROR METADATA ===\n")
print(response.get("metadata"))