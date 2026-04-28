import re
from dataclasses import asdict
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rag.schemas import ValidationResult, SentenceValidationItem


CITATION_PATTERN = re.compile(r"\[Fuente\s+(\d+)\]", re.IGNORECASE)


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def split_claims(answer: str) -> List[str]:
    """
    Divide la respuesta en unidades validables.
    Prioriza viñetas. Si no hay viñetas, divide por oraciones simples.
    """
    lines = [
        clean_text(line.lstrip("-•* ").strip())
        for line in answer.splitlines()
        if clean_text(line.lstrip("-•* ").strip())
    ]

    if len(lines) > 1:
        return lines

    # fallback simple por oraciones
    sentences = re.split(r"(?<=[.!?])\s+", clean_text(answer))
    return [s for s in sentences if s]


def extract_source_numbers(claim: str) -> List[int]:
    """
    Extrae citas tipo [Fuente 1], [Fuente 2], etc.
    """
    return [int(match) for match in CITATION_PATTERN.findall(claim)]


def remove_citations(text: str) -> str:
    return clean_text(CITATION_PATTERN.sub("", text))


def get_cited_context(
    source_numbers: List[int],
    chunks: List[dict],
) -> Tuple[str, List[str]]:
    """
    Mapea [Fuente N] al chunk N-1, porque el prompt enumera desde 1.
    """
    cited_texts = []
    cited_chunk_ids = []

    for n in source_numbers:
        idx = n - 1
        if 0 <= idx < len(chunks):
            ch = chunks[idx]
            cited_texts.append(ch.get("text", ""))
            cited_chunk_ids.append(ch.get("chunk_id", "?"))

    return "\n\n".join(cited_texts), cited_chunk_ids


def tfidf_similarity(a: str, b: str) -> float:
    """
    Similitud TF-IDF entre afirmación y contexto citado.
    Usa n-gramas de caracteres para tolerar variaciones leves.
    """
    a = clean_text(a)
    b = clean_text(b)

    if not a or not b:
        return 0.0

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
    )

    try:
        matrix = vectorizer.fit_transform([a, b])
        sim = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        return float(sim)
    except ValueError:
        return 0.0


def validate_sentences(
    answer: str,
    chunks: list,
    threshold: float = 0.22,
    partial_threshold: float = 0.01,
    invalid_threshold: float = 0.5,
):
    """
    Valida una respuesta con citas obligatorias.

    - Divide respuesta en afirmaciones/viñetas.
    - Verifica que cada afirmación tenga al menos una cita válida.
    - Compara cada afirmación contra los chunks citados.
    - Calcula USR: proporción de afirmaciones no soportadas.
    """

    claims = split_claims(answer)

    if not claims:
        return ValidationResult(
            usr=1.0,
            unsupported_sentences=[],
            sentence_results=[],
            citation_consistency=0.0,
            is_valid=False,
            decision="invalidated",
        )

    sentence_results = []
    unsupported = []
    citation_valid_count = 0

    for claim in claims:
        source_numbers = extract_source_numbers(claim)
        claim_without_citations = remove_citations(claim)

        cited_context, cited_chunk_ids = get_cited_context(source_numbers, chunks)

        has_valid_citation = bool(cited_context.strip())

        if has_valid_citation:
            citation_valid_count += 1

        similarity = tfidf_similarity(claim_without_citations, cited_context)

        supported = has_valid_citation and similarity >= threshold

        reason = None
        if not source_numbers:
            reason = "La afirmación no contiene citas."
        elif not has_valid_citation:
            reason = "La afirmación contiene citas inválidas."
        elif similarity < threshold:
            reason = "La similitud con el contexto citado está por debajo del umbral."

        if not supported:
            unsupported.append(claim)

        best_chunk_id = cited_chunk_ids[0] if cited_chunk_ids else None

        sentence_results.append(
            SentenceValidationItem(
                sentence=claim,
                supported=supported,
                max_similarity=similarity,
                best_chunk_id=best_chunk_id,
                reason=reason,
            )
        )

    usr = len(unsupported) / len(claims)
    citation_consistency = citation_valid_count / len(claims)

    if usr >= invalid_threshold:
        decision = "invalidated"
        is_valid = False
    elif usr >= partial_threshold:
        decision = "partial"
        is_valid = True
    else:
        decision = "answered"
        is_valid = True

    return ValidationResult(
        usr=usr,
        unsupported_sentences=unsupported,
        sentence_results=sentence_results,
        citation_consistency=citation_consistency,
        is_valid=is_valid,
        decision=decision,
    )