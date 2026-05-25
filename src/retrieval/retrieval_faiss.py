import json
import os

import faiss
import numpy as np
from openai import OpenAI
import tiktoken

from .medspaner_bridge import run_medspaner_question
from .utils_env import get_openai_api_key


# CONFIG

api_key = get_openai_api_key()
client = OpenAI(api_key=api_key)

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 350
TOP_K = 5

BASE_VECTOR_DIR = "../vector_index"
DEFAULT_INDEX_VARIANT = "sections"


def get_index_paths(index_variant: str = DEFAULT_INDEX_VARIANT) -> dict:
    """
    Devuelve las rutas del índice según la variante experimental.

    Variantes esperadas:
    - flat
    - sections
    """
    base_dir = os.path.join(BASE_VECTOR_DIR, index_variant)

    return {
        "index": os.path.join(base_dir, "index.faiss"),
        "metadata": os.path.join(base_dir, "metadata.json"),
        "mapping": os.path.join(base_dir, "mapping.json"),
    }


# Utilidades

def clip_text(texto: str) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(texto)

    if len(tokens) <= MAX_TOKENS:
        return texto

    return enc.decode(tokens[:MAX_TOKENS])


def embed(text: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return np.array(resp.data[0].embedding, dtype="float32")


def load_faiss(index_variant: str = DEFAULT_INDEX_VARIANT):
    """
    Carga índice FAISS y metadata según variante:
    vector_index/flat/
    vector_index/sections/
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
    index_variant: str = DEFAULT_INDEX_VARIANT,
):
    """
    Retrieval semántico base con FAISS.

    Parámetros:
    - query_text: pregunta del usuario.
    - top_k: número de chunks a recuperar.
    - index_variant: índice experimental a usar:
        - "flat"
        - "sections"

    Retorna:
    - chunks refinados
    - signals extraídas de la consulta
    - salida cruda MEDSPANER
    """

    # 1. Análisis médico con MEDSPANER
    medspaner_output = run_medspaner_question(query_text)
    signals = extract_query_signals(medspaner_output)

    # 2. Embedding de la pregunta
    query_emb = embed(query_text)
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

        candidates.append(meta)

    # 6. Filtrado + reranking por entidades de MEDSPANER
    refined = filter_by_medical_signals(candidates, signals)

    return refined[:top_k], signals, medspaner_output