from openai import OpenAI
from rag import default_rag_client


QUESTIONS = [
    "¿Para qué sirve el paracetamol?",
    "¿Cuál es la dosis recomendada de paracetamol?",
    "¿Qué contraindicaciones tiene el paracetamol?",
    "¿Qué efectos adversos puede causar el paracetamol?",
    "¿Se puede tomar paracetamol durante la lactancia?",
    "¿Qué ocurre si se toma más paracetamol del recomendado?",
    "¿Puede tomarse paracetamol con alcohol?",
    "¿Qué debe hacer si olvida tomar paracetamol?",
]

EXPERIMENTS = ["baseline", "p1_retrieval"]


def compact_chunk(ch: dict) -> str:
    text = ch.get("text", "").replace("\n", " ").strip()
    preview = text[:90] + ("..." if len(text) > 90 else "")

    sources = ch.get("rrf_sources")
    if sources:
        source_info = f"rrf_sources={sources}"
    else:
        source_info = "rrf_sources=None"

    return (
        f"{ch.get('chunk_id')} | "
        f"score={round(ch.get('score', 0), 4)} | "
        f"faiss_rank={ch.get('faiss_rank')} | "
        f"rrf_rank={ch.get('rrf_rank')} | "
        f"{source_info} | "
        f"{preview}"
    )


def main():
    client = OpenAI()

    rag_clients = {
        exp: default_rag_client(client, experiment=exp)
        for exp in EXPERIMENTS
    }

    for idx, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 120)
        print(f"Q{idx}: {question}")
        print("=" * 120)

        for exp in EXPERIMENTS:
            response = rag_clients[exp].query(question)

            print(f"\n[{exp}]")
            print(f"STATUS: {response.get('status')}")
            print(f"ANSWER: {response.get('answer')}")

            print("CHUNKS:")
            for ch in response.get("chunks", []):
                print(" - " + compact_chunk(ch))

            error = response.get("metadata", {}).get("error")
            if error:
                print(f"ERROR: {error}")


if __name__ == "__main__":
    main()