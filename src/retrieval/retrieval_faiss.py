import json
import os

import faiss
import numpy as np
import tiktoken

from embeddings.factory import get_embedder
from .medspaner_bridge import run_medspaner_question


# CONFIG

MAX_TOKENS = 350
TOP_K = 5

BASE_VECTOR_DIR = "../vector_index"

DEFAULT_CHUNKING_VARIANT = "sections"
DEFAULT_EMBEDDING_MODEL = "openai"


def normalize_embedding_model_name(name: str) -> str:
    name = name.lower().strip()

    aliases = {
        "openai": "openai",
        "e5": "multilingual_e5",
        "multilingual-e5": "multilingual_e5",
        "multilingual_e5": "multilingual_e5",
        "medcpt": "medcpt",
    }

    if name not in aliases:
        raise ValueError(
            f"embedding_model no soportado: {name}. "
            f"Opciones: openai, multilingual_e5, medcpt"
        )

    return aliases[name]


def normalize_chunking_variant(name: str) -> str:
    name = name.lower().strip()

    if name not in {"flat", "sections"}:
        raise ValueError(
            f"chunking_variant no soportado: {name}. "
            f"Opciones: flat, sections"
        )

    return name


def build_index_variant(
    chunking_variant: str = DEFAULT_CHUNKING_VARIANT,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    index_suffix: str | None = None,
) -> str:
    chunking_variant = normalize_chunking_variant(chunking_variant)
    embedding_model = normalize_embedding_model_name(embedding_model)

    index_variant = f"{chunking_variant}_{embedding_model}"

    if index_suffix:
        index_variant = f"{index_variant}_{index_suffix}"

    return index_variant


def get_index_paths(index_variant: str) -> dict:
    base_dir = os.path.join(BASE_VECTOR_DIR, index_variant)

    return {
        "index": os.path.join(base_dir, "index.faiss"),
        "metadata": os.path.join(base_dir, "metadata.json"),
    }


# Utilidades

def clip_text(texto: str) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(texto)

    if len(tokens) <= MAX_TOKENS:
        return texto

    return enc.decode(tokens[:MAX_TOKENS])


def load_faiss(index_variant: str):
    """
    Carga índice FAISS y metadata según variante:
    - sections_openai
    - sections_multilingual_e5
    - sections_medcpt
    - flat_openai
    - flat_multilingual_e5
    - flat_medcpt
    """
    paths = get_index_paths(index_variant)

    if not os.path.exists(paths["index"]):
        raise FileNotFoundError(
            f"No existe index.faiss para index_variant='{index_variant}' en: {paths['index']}"
        )

    if not os.path.exists(paths["metadata"]):
        raise FileNotFoundError(
            f"No existe metadata.json para index_variant='{index_variant}' en: {paths['metadata']}"
        )

    index = faiss.read_index(paths["index"])

    with open(paths["metadata"], "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return index, metadata


def embed_query(
    text: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray:
    embedding_model = normalize_embedding_model_name(embedding_model)

    embedder = get_embedder(embedding_model)

    return embedder.embed_query(text)


# Interpretación de entidades MEDSPANER

def extract_query_signals(entities):
    """
    A partir del JSON de MEDSPANER en la consulta, detectamos:
    - medicamentos (CHEM)
    - dosis (Dose/Strength)
    - patologías (DISO)
    - vías o formas farmacéuticas (Form/Route)
    """

    signals = {
        "meds": [],
        "doses": [],
        "diso": [],
        "forms": [],
    }

    if not isinstance(entities, list):
        return signals

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        label = ent.get("entity_group")
        word = ent.get("word", "").lower()

        if label == "CHEM":
            signals["meds"].append(word)

        elif label in ("Dose", "Strength"):
            signals["doses"].append(word)

        elif label == "DISO":
            signals["diso"].append(word)

        elif label in ("Form", "Route"):
            signals["forms"].append(word)

    return signals


# Filtrado estructurado por entidades

def filter_by_medical_signals(candidates, signals):
    """
    Si la consulta menciona medicamento, dosis, patología, etc.,
    priorizamos chunks que contengan entidades concordantes.
    """

    meds_q = signals.get("meds", [])
    diso_q = signals.get("diso", [])
    forms_q = signals.get("forms", [])

    if not meds_q and not diso_q and not forms_q:
        return candidates

    filtered = []

    for c in candidates:
        chunk_text = c.get("text", "").lower()

        score = 0

        for m in meds_q:
            if m in chunk_text:
                score += 2

        for d in diso_q:
            if d in chunk_text:
                score += 1

        for f in forms_q:
            if f in chunk_text:
                score += 1

        c["_rerank_score"] = score
        filtered.append(c)

    filtered.sort(
        key=lambda x: (
            x.get("_rerank_score", 0),
            x.get("score", 0.0),
        ),
        reverse=True,
    )

    return filtered


# RETRIEVAL PRINCIPAL

def retrieve_chunks(
    query_text: str,
    top_k: int = TOP_K,
    chunking_variant: str = DEFAULT_CHUNKING_VARIANT,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    index_suffix: str | None = None,
):
    """
    Retrieval semántico base con FAISS.

    Parámetros:
    - query_text: pregunta del usuario.
    - top_k: número de chunks a recuperar.
    - chunking_variant: flat o sections.
    - embedding_model: openai, multilingual_e5 o medcpt.
    - index_suffix: sufijo opcional para cargar índices alternativos, por ejemplo "corrupted".
    Retorna:
    - chunks refinados
    - signals extraídas de la consulta
    - salida cruda MEDSPANER
    """

    chunking_variant = normalize_chunking_variant(chunking_variant)
    embedding_model = normalize_embedding_model_name(embedding_model)

    index_variant = build_index_variant(
        chunking_variant=chunking_variant,
        embedding_model=embedding_model,
        index_suffix=index_suffix,
    )

    # 1. Análisis médico con MEDSPANER
    medspaner_output = run_medspaner_question(query_text)
    signals = extract_query_signals(medspaner_output)

    # 2. Embedding de la pregunta usando el mismo modelo del índice
    query_emb = embed_query(
        query_text,
        embedding_model=embedding_model,
    )

    query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)

    # 3. Cargar FAISS y metadata según variante
    index, metadata = load_faiss(index_variant=index_variant)

    # 4. Similaridad vectorial
    scores, idxs = index.search(query_emb, top_k)

    # 5. Construir lista de candidatos
    candidates = []

    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue

        if idx >= len(metadata):
            continue

        meta = dict(metadata[idx])
        meta["score"] = float(score)
        meta["index_variant"] = meta.get("index_variant", index_variant)
        meta["chunking_strategy"] = meta.get("chunking_strategy", chunking_variant)
        meta["embedding_model"] = meta.get("embedding_model", embedding_model)

        candidates.append(meta)

    # 6. Filtrado + reranking por entidades de MEDSPANER
    refined = filter_by_medical_signals(candidates, signals)

    return refined[:top_k], signals, medspaner_output