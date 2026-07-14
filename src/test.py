from openai import OpenAI
from rag import default_rag_client
import json

client = OpenAI()

rag = default_rag_client(
    client,
    experiment="p2_openai_knowledge_noise_no_validation"
)

response = rag.query(
    "En alcohólicos crónicos, ¿cuántos g/día de paracetamol como máximo se deben tomar?"
)

print("ANSWER:")
print(response["answer"])

print("\nSTATUS:")
print(response["status"])

print("\nCHUNKS:")
for ch in response["chunks"]:
    print(ch["uid"])
    print(ch["text"][:500])
    print("---")

print("\nVALIDATION:")
print(json.dumps(response["validation"], ensure_ascii=False, indent=2))