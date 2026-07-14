import json
from openai import OpenAI
from rag import default_rag_client


QUESTION = "¿Para qué sirve el paracetamol?"

EXPERIMENTS = [
    {
        "name": "baseline",
        "title": "1. Baseline: FAISS + generación directa",
        "description": (
            "Sistema base. Recupera chunks con FAISS y genera una respuesta "
            "sin citas ni validación post-generación."
        ),
    },
    {
        "name": "p1_retrieval",
        "title": "2. Propuesta 1 - Paso A: Retrieval híbrido",
        "description": (
            "Añade BM25 + RRF sobre FAISS. Busca combinar recuperación semántica "
            "y léxica para mejorar cobertura."
        ),
    },
    {
        "name": "p1_citations",
        "title": "3. Propuesta 1 - Paso B: Generación con citas",
        "description": (
            "Mantiene retrieval híbrido y modifica el prompt para obligar al modelo "
            "a citar las fuentes usadas en cada afirmación."
        ),
    },
    {
        "name": "propuesta_1_full",
        "title": "4. Propuesta 1 completa: citas + validación por oración",
        "description": (
            "Añade validación post-generación. Calcula USR y consistencia de citas "
            "para verificar si la respuesta está soportada por el contexto."
        ),
    },
]


def line(char="=", size=110):
    print(char * size)


def compact_chunk(ch: dict) -> str:
    chunk_id = ch.get("chunk_id")
    doc = ch.get("document_id")
    score = ch.get("score")
    rrf_rank = ch.get("rrf_rank")
    rrf_sources = ch.get("rrf_sources")

    text = " ".join((ch.get("text") or "").split())
    preview = text[:140] + ("..." if len(text) > 140 else "")

    if rrf_rank:
        rank_info = f"rrf_rank={rrf_rank}, sources={rrf_sources}"
    else:
        rank_info = "baseline_faiss"

    return (
        f"- {doc}/{chunk_id} | score={score} | {rank_info}\n"
        f"  preview: {preview}"
    )


def print_chunks(chunks: list[dict]):
    print("\nChunks recuperados:")
    for ch in chunks:
        print(compact_chunk(ch))


def print_validation(validation: dict | None):
    print("\nValidación:")

    if not validation:
        print("- No aplica en esta configuración.")
        return

    print(f"- USR: {validation.get('usr')}")
    print(f"- Citation consistency: {validation.get('citation_consistency')}")
    print(f"- Decisión: {validation.get('decision')}")
    print(f"- Oraciones no soportadas: {len(validation.get('unsupported_sentences', []))}")

    sentence_results = validation.get("sentence_results", [])

    if sentence_results:
        print("\nDetalle por afirmación:")
        for idx, item in enumerate(sentence_results, start=1):
            print(
                f"  {idx}. supported={item.get('supported')} | "
                f"sim={round(item.get('max_similarity', 0), 4)} | "
                f"best_chunk={item.get('best_chunk_id')}"
            )
            print(f"     {item.get('sentence')}")


def print_response(response: dict):
    print("\nEstado:")
    print(f"- {response.get('status')}")

    print("\nRespuesta:")
    print(response.get("answer"))

    print_chunks(response.get("chunks", []))
    print_validation(response.get("validation"))

    error = response.get("metadata", {}).get("error")
    if error:
        print("\nError:")
        print(error)


def main():
    client = OpenAI()

    line()
    print("DEMO PROGRESIVA - PROPUESTA 1")
    line()
    print(f"Pregunta usada: {QUESTION}")
    print(
        "\nObjetivo: mostrar cómo evoluciona el sistema desde el baseline "
        "hasta la Propuesta 1 completa."
    )

    for exp in EXPERIMENTS:
        line()
        print(exp["title"])
        line("-")
        print(exp["description"])

        rag = default_rag_client(
            client,
            experiment=exp["name"],
            logdir="evals/logs/demo_progress",
        )

        response = rag.query(QUESTION)
        print_response(response)

    line()
    print("RESUMEN FINAL")
    line()
    print(
        "La Propuesta 1 incorpora progresivamente retrieval híbrido, generación "
        "con citas obligatorias y validación por oración. Esto permite pasar de "
        "un sistema que solo responde a un sistema que recupera, cita y valida "
        "sus afirmaciones contra el contexto."
    )


if __name__ == "__main__":
    main()