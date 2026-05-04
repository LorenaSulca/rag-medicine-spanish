from retrieval.hybrid_retriever import retrieve_hybrid
from retrieval.retrieval_faiss import (
    extract_query_signals,
    TOP_K,
)
from retrieval.medspaner_bridge import run_medspaner_question


def infer_query_complexity(question: str, signals: dict) -> dict:
    """
    Calcula una complejidad simple de la consulta para decidir cuántos chunks recuperar.

    La idea es evitar un K fijo:
    - preguntas simples: menos chunks
    - preguntas con varias entidades o varias intenciones: más chunks
    """

    q = question.lower()

    entity_count = (
        len(signals.get("meds", []))
        + len(signals.get("doses", []))
        + len(signals.get("diso", []))
        + len(signals.get("forms", []))
    )

    intent_terms = {
        "dose": [
            "dosis",
            "cuánto",
            "cuanta",
            "cuántos",
            "cuantas",
            "mg",
            "gramos",
            "comprimidos",
        ],
        "contraindications": [
            "contraindicaciones",
            "contraindicado",
            "no tome",
            "no debe",
            "alérgico",
            "alergia",
        ],
        "adverse_effects": [
            "efectos adversos",
            "efectos secundarios",
            "reacciones adversas",
            "puede causar",
        ],
        "interactions": [
            "interacciones",
            "interactúa",
            "interactua",
            "junto con",
            "alcohol",
            "otros medicamentos",
        ],
        "pregnancy_lactation": [
            "embarazo",
            "lactancia",
            "fertilidad",
            "leche materna",
        ],
        "overdose": [
            "sobredosis",
            "más de la dosis",
            "más paracetamol",
            "más medicamento",
            "más recomendado",
        ],
        "missed_dose": [
            "olvidó",
            "olvido",
            "olvidar",
            "dosis olvidada",
        ],
    }

    matched_intents = []

    for intent, terms in intent_terms.items():
        if any(term in q for term in terms):
            matched_intents.append(intent)

    # Score simple, entendible y reportable en metodología.
    complexity_score = 0
    complexity_score += min(entity_count, 4)
    complexity_score += len(matched_intents)

    if len(question.split()) > 15:
        complexity_score += 1

    if any(connector in q for connector in [" y ", " además", "también", " o "]):
        complexity_score += 1

    if complexity_score <= 1:
        level = "low"
    elif complexity_score <= 3:
        level = "medium"
    else:
        level = "high"

    return {
        "level": level,
        "score": complexity_score,
        "entity_count": entity_count,
        "matched_intents": matched_intents,
    }


def choose_dynamic_k(complexity: dict) -> int:
    """
    Traduce la complejidad de la pregunta a un K dinámico.
    """

    level = complexity["level"]

    if level == "low":
        return 3

    if level == "medium":
        return 5

    return 8


def retrieve_dynamic(
    query_text: str,
    min_k: int = 3,
    default_k: int = TOP_K,
    max_k: int = 8,
    candidate_k: int = 12,
):
    """
    Retrieval dinámico para Propuesta 2.

    Flujo:
    1. Analiza la pregunta con MEDSPANER.
    2. Estima complejidad de consulta.
    3. Decide K dinámico.
    4. Ejecuta retrieval híbrido usando ese K.
    """

    medspaner_output = run_medspaner_question(query_text)
    signals = extract_query_signals(medspaner_output)

    complexity = infer_query_complexity(query_text, signals)

    dynamic_k = choose_dynamic_k(complexity)
    dynamic_k = max(min_k, min(dynamic_k, max_k))

    chunks, _, _ = retrieve_hybrid(
        query_text=query_text,
        top_k=dynamic_k,
        candidate_k=max(candidate_k, dynamic_k),
        dynamic_k=False,
    )

    for ch in chunks:
        ch["dynamic_k"] = dynamic_k
        ch["query_complexity"] = complexity

    return chunks, signals, medspaner_output