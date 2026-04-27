from openai import OpenAI
from rag import default_rag_client

client = OpenAI()
question = "¿Para qué sirve el paracetamol?"

for exp in ["baseline", "p1_retrieval"]:
    rag = default_rag_client(client, experiment=exp)
    response = rag.query(question)

    print(f"\n\n=== EXPERIMENTO: {exp} ===")
    print("\nANSWER:")
    print(response["answer"])
    print("\nCHUNKS:")
    for ch in response["chunks"]:
        print({
            "chunk_id": ch.get("chunk_id"),
            "score": ch.get("score"),
            "faiss_rank": ch.get("faiss_rank"),
            "rrf_rank": ch.get("rrf_rank"),
            "rrf_sources": ch.get("rrf_sources"),
            "_rerank_score": ch.get("_rerank_score"),
        })
        print(ch.get("text", "")[:150], "\n")