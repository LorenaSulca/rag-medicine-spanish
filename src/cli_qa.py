# cli_qa.py

from answer_generator import answer_question

def main():
    print("RAG Prospectos Medicamentos español")
    print("Escribe una pregunta sobre un medicamento (o 'salir')\n")

    while True:
        q = input("Pregunta: ").strip()
        if not q:
            continue
        if q.lower() in ("salir", "exit", "q"):
            break

        result = answer_question(q)

        print("\nRESPUESTA\n")
        print(result["answer"])
        print("\n--- CONTEXTO USADO ---\n")

        for i, ch in enumerate(result["chunks"], start=1):
            print(f"[Fuente {i} | doc: {ch.get('document_id')} | chunk: {ch.get('chunk_id')} | score: {round(ch.get('score', 0.0),3)}]")
            print(ch.get("text", "").strip())
            print("")

if __name__ == "__main__":
    main()
