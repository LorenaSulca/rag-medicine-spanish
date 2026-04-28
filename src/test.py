from openai import OpenAI
from rag import default_rag_client
import json

client = OpenAI()

rag = default_rag_client(
    client,
    experiment="propuesta_1_full"
)

response = rag.query("¿Para qué sirve el paracetamol?")

print(response["answer"])
print(response["status"])

print("\nVALIDATION:")
print(json.dumps(response["validation"], ensure_ascii=False, indent=2))