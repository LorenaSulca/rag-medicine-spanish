from __future__ import annotations

import random
from typing import Iterable


def normalize_text(value: str | None) -> str:
    if not value:
        return ""

    return " ".join(value.lower().strip().split())


def get_chunk_uid(chunk: dict) -> str:
    """
    Devuelve un identificador estable para comparar chunks.
    """

    return (
        chunk.get("uid")
        or f"{chunk.get('document_id', '')}_{chunk.get('chunk_id', '')}"
    )


def get_retrieved_uids(chunks: Iterable[dict]) -> set[str]:
    return {
        get_chunk_uid(chunk)
        for chunk in chunks
    }


def get_retrieved_sections(chunks: Iterable[dict]) -> set[str]:
    return {
        normalize_text(chunk.get("section_name"))
        for chunk in chunks
        if chunk.get("section_name")
    }


def score_noise_candidate(
    candidate: dict,
    retrieved_sections: set[str],
) -> int:
    """
    Puntúa candidatos de ruido.

    La idea es favorecer distractores semánticos plausibles:
    - mismo documento médico,
    - sección distinta a las ya recuperadas,
    - con contenido biomédico o entidades.

    No se usa como score de relevancia, sino como heurística para elegir
    ruido plausible pero no idéntico al contexto recuperado.
    """

    score = 0

    section = normalize_text(candidate.get("section_name"))

    if section and section not in retrieved_sections:
        score += 2

    entities = candidate.get("entities") or {}

    if isinstance(entities, dict) and any(entities.values()):
        score += 1

    text = normalize_text(candidate.get("text"))

    biomedical_terms = [
        "dosis",
        "sobredosis",
        "contraindic",
        "efectos adversos",
        "reacciones adversas",
        "interacciones",
        "embarazo",
        "lactancia",
        "medicamento",
        "comprimido",
        "paracetamol",
    ]

    if any(term in text for term in biomedical_terms):
        score += 1

    return score


def select_semantic_distractors(
    retrieved_chunks: list[dict],
    metadata: list[dict],
    noise_chunks: int = 2,
    seed: int = 42,
) -> list[dict]:
    """
    Selecciona distractores semánticos desde la misma metadata del índice.

    Criterios:
    - excluir chunks ya recuperados;
    - preferir chunks del mismo documento si es posible;
    - preferir secciones distintas a las recuperadas;
    - preferir chunks con señales biomédicas.

    Esto implementa ruido semánticamente plausible, no ruido aleatorio puro.
    """

    if noise_chunks <= 0:
        return []

    rng = random.Random(seed)

    retrieved_uids = get_retrieved_uids(retrieved_chunks)
    retrieved_sections = get_retrieved_sections(retrieved_chunks)

    retrieved_doc_ids = {
        ch.get("document_id")
        for ch in retrieved_chunks
        if ch.get("document_id")
    }

    candidates = []

    for item in metadata:
        uid = get_chunk_uid(item)

        if uid in retrieved_uids:
            continue

        same_document = (
            item.get("document_id") in retrieved_doc_ids
            if retrieved_doc_ids
            else True
        )

        if not same_document:
            continue

        candidate = dict(item)
        candidate["_noise_candidate_score"] = score_noise_candidate(
            candidate,
            retrieved_sections,
        )

        candidates.append(candidate)

    if not candidates:
        return []

    candidates.sort(
        key=lambda x: (
            x.get("_noise_candidate_score", 0),
            rng.random(),
        ),
        reverse=True,
    )

    selected = candidates[:noise_chunks]

    distractors = []

    for idx, item in enumerate(selected, start=1):
        noise = dict(item)

        noise["_is_noise"] = True
        noise["_noise_type"] = "semantic_distractor"
        noise["_noise_rank"] = idx
        noise["_retrieval_source"] = "noise"

        if "score" not in noise:
            noise["score"] = 0.0

        distractors.append(noise)

    return distractors


def inject_noise_chunks(
    retrieved_chunks: list[dict],
    metadata: list[dict],
    noise_chunks: int = 2,
    seed: int = 42,
    placement: str = "end",
) -> list[dict]:
    """
    Inserta distractores en el contexto recuperado.

    placement:
    - end: añade ruido al final
    - middle: inserta ruido en el medio
    - interleave: intercala ruido entre chunks recuperados
    """

    distractors = select_semantic_distractors(
        retrieved_chunks=retrieved_chunks,
        metadata=metadata,
        noise_chunks=noise_chunks,
        seed=seed,
    )

    if not distractors:
        return retrieved_chunks

    clean_chunks = [
        dict(ch, _is_noise=ch.get("_is_noise", False))
        for ch in retrieved_chunks
    ]

    if placement == "end":
        return clean_chunks + distractors

    if placement == "middle":
        midpoint = len(clean_chunks) // 2
        return (
            clean_chunks[:midpoint]
            + distractors
            + clean_chunks[midpoint:]
        )

    if placement == "interleave":
        output = []
        max_len = max(len(clean_chunks), len(distractors))

        for i in range(max_len):
            if i < len(clean_chunks):
                output.append(clean_chunks[i])

            if i < len(distractors):
                output.append(distractors[i])

        return output

    raise ValueError(
        f"placement no soportado: {placement}. "
        f"Opciones: end, middle, interleave"
    )