from openai import OpenAI
from rag import default_rag_client

client = OpenAI()

rag = default_rag_client(client)

response = rag.query("¿Para qué sirve el paracetamol?")

print("\n=== ANSWER ===\n")
print(response["answer"])

print("\n=== STATUS ===\n")
print(response["status"])

print("\n=== CONTEXTS ===\n")
for c in response["contexts"]:
    print("-", c[:120])