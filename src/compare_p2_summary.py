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
    "¿Cuál es la dosis y qué ocurre si se toma más paracetamol del recomendado?",
]

EXPERIMENTS = [
    "propuesta_1_full",
    "p2_dynamic_retrieval",
    "p2_refine",
    "propuesta_2_full",
]


def trim(text: str, limit: int = 420) -> str:
    clean = " ".join((text or "").split())
    return clean[:limit] + ("..." if len(clean) > limit else "")


def compact_chunks(chunks: list[dict]) -> str:
    parts = []

    for ch in chunks:
        chunk_id = ch.get("chunk_id")
        dynamic_k = ch.get("dynamic_k")
        complexity = ch.get("query_complexity")
        rrf_rank = ch.get("rrf_rank")

        if dynamic_k:
            level = complexity.get("level") if isinstance(complexity, dict) else None
            intents = complexity.get("matched_intents") if isinstance(complexity, dict) else None
            parts.append(f"{chunk_id}(rrf={rrf_rank},k={dynamic_k},level={level},intents={intents})")
        else:
            parts.append(f"{chunk_id}(rrf={rrf_rank})")

    return ", ".join(parts)


def compact_validation(validation: dict | None) -> str:
    if not validation:
        return "validation=None"

    multilevel = validation.get("multilevel") or {}

    return (
        f"usr={validation.get('usr')}, "
        f"cit={validation.get('citation_consistency')}, "
        f"unsupported={len(validation.get('unsupported_sentences', []))}, "
        f"numeric_rate={multilevel.get('numeric_support_rate')}"
    )


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
            print(f"ANSWER: {trim(response.get('answer'))}")
            print(f"CHUNKS: {compact_chunks(response.get('chunks', []))}")
            print(f"VALIDATION: {compact_validation(response.get('validation'))}")

            error = response.get("metadata", {}).get("error")
            if error:
                print(f"ERROR: {error}")


if __name__ == "__main__":
    main()