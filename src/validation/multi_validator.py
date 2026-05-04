import re
from typing import List, Dict, Any

from validation.sentence_validator import (
    validate_sentences,
    split_claims,
    extract_source_numbers,
    remove_citations,
    get_cited_context,
)
from rag.schemas import ValidationResult


NUMBER_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|g|gramos|comprimidos|horas|h|días|dias|veces)\b",
    re.IGNORECASE,
)


def extract_numeric_medical_mentions(text: str) -> List[str]:
    """
    Extrae menciones numéricas médicamente relevantes.
    Ejemplos:
    - 1 g
    - 4 g
    - 24 horas
    - 3-4 veces  (este caso se captura parcialmente como 4 veces si aparece separado)
    """
    if not text:
        return []

    normalized = text.lower().replace(",", ".")
    return list(set(match.group(0).strip() for match in NUMBER_PATTERN.finditer(normalized)))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def validate_numeric_mentions_against_citations(
    answer: str,
    chunks: List[dict],
) -> Dict[str, Any]:
    """
    Verifica si las menciones numéricas/dosis de cada afirmación aparecen
    en el contexto citado por esa afirmación.

    Esto no reemplaza el USR; lo complementa.
    """

    claims = split_claims(answer)
    results = []
    unsupported_mentions = []

    for claim in claims:
        source_numbers = extract_source_numbers(claim)
        claim_text = remove_citations(claim)

        mentions = extract_numeric_medical_mentions(claim_text)
        cited_context, cited_chunk_ids = get_cited_context(source_numbers, chunks)
        cited_context_norm = normalize_text(cited_context)

        claim_result = {
            "claim": claim,
            "cited_chunk_ids": cited_chunk_ids,
            "mentions": [],
        }

        for mention in mentions:
            mention_norm = normalize_text(mention)

            supported = mention_norm in cited_context_norm

            item = {
                "mention": mention,
                "supported": supported,
            }

            claim_result["mentions"].append(item)

            if not supported:
                unsupported_mentions.append({
                    "claim": claim,
                    "mention": mention,
                    "cited_chunk_ids": cited_chunk_ids,
                })

        results.append(claim_result)

    total_mentions = sum(len(r["mentions"]) for r in results)
    unsupported_count = len(unsupported_mentions)

    if total_mentions == 0:
        numeric_support_rate = None
    else:
        numeric_support_rate = 1 - (unsupported_count / total_mentions)

    return {
        "numeric_support_rate": numeric_support_rate,
        "unsupported_numeric_mentions": unsupported_mentions,
        "numeric_validation_results": results,
    }


def validate_multilevel(
    answer: str,
    chunks: list,
    threshold: float = 0.20,
    partial_threshold: float = 0.01,
    invalid_threshold: float = 0.50,
) -> ValidationResult:
    """
    Validación multinivel inicial.

    Nivel 1:
    - Validación por oración / USR.

    Nivel 2:
    - Consistencia de citas.

    Nivel 3:
    - Verificación simple de menciones numéricas médicas contra el contexto citado.

    Retorna ValidationResult extendiendo metadata dentro de sentence_results vía campos extra
    no soportados por dataclass, por eso añadimos la información en atributos dinámicos.
    """

    base_validation = validate_sentences(
        answer=answer,
        chunks=chunks,
        threshold=threshold,
        partial_threshold=partial_threshold,
        invalid_threshold=invalid_threshold,
    )

    numeric_validation = validate_numeric_mentions_against_citations(
        answer=answer,
        chunks=chunks,
    )

    unsupported_numeric = numeric_validation["unsupported_numeric_mentions"]

    # Si hay menciones numéricas no soportadas, endurecemos la decisión.
    if unsupported_numeric:
        base_validation.is_valid = False

        if base_validation.decision == "answered":
            base_validation.decision = "partial"

        # Añadimos al listado de unsupported para que sea visible.
        for item in unsupported_numeric:
            base_validation.unsupported_sentences.append(
                f"Mención numérica no soportada: '{item['mention']}' en: {item['claim']}"
            )

        # Recalcular USR aproximado si se agregaron fallos numéricos.
        claims = split_claims(answer)
        if claims:
            unsupported_claims = set(base_validation.unsupported_sentences)
            base_validation.usr = min(
                1.0,
                max(base_validation.usr, len(unsupported_numeric) / len(claims))
            )

    # Guardamos información extra como atributo dinámico.
    # Luego el pipeline debe serializarlo manualmente si queremos verlo en response.
    base_validation.multilevel = {
        "numeric_support_rate": numeric_validation["numeric_support_rate"],
        "unsupported_numeric_mentions": numeric_validation["unsupported_numeric_mentions"],
        "numeric_validation_results": numeric_validation["numeric_validation_results"],
    }

    return base_validation