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

EXPERIMENTS = [
    "baseline",
    "p1_retrieval",
    "p1_citations",
    "propuesta_1_full",
]


def compact_chunks(chunks: list[dict]) -> str:
    ids = []
    for ch in chunks:
        chunk_id = ch.get("chunk_id")
        rrf_rank = ch.get("rrf_rank")
        sources = ch.get("rrf_sources")

        if rrf_rank:
            ids.append(f"{chunk_id}(rrf={rrf_rank},src={sources})")
        else:
            ids.append(str(chunk_id))

    return ", ".join(ids)


def compact_validation(validation: dict | None) -> str:
    if not validation:
        return "validation=None"

    usr = validation.get("usr")
    citation = validation.get("citation_consistency")
    unsupported = validation.get("unsupported_sentences", [])

    return (
        f"usr={usr}, "
        f"citation_consistency={citation}, "
        f"unsupported_count={len(unsupported)}"
    )


def trim_answer(answer: str, limit: int = 500) -> str:
    clean = " ".join((answer or "").split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def main():
    client = OpenAI()

    clients = {
        exp: default_rag_client(client, experiment=exp)
        for exp in EXPERIMENTS
    }

    for i, question in enumerate(QUESTIONS, start=1):
        print("\n" + "=" * 120)
        print(f"Q{i}: {question}")
        print("=" * 120)

        for exp in EXPERIMENTS:
            response = clients[exp].query(question)

            print(f"\n[{exp}]")
            print(f"STATUS: {response.get('status')}")
            print(f"ANSWER: {trim_answer(response.get('answer'))}")
            print(f"CHUNKS: {compact_chunks(response.get('chunks', []))}")
            print(f"VALIDATION: {compact_validation(response.get('validation'))}")

            error = response.get("metadata", {}).get("error")
            if error:
                print(f"ERROR: {error}")


if __name__ == "__main__":
    main()