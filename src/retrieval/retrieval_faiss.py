import json
import numpy as np
import faiss
from openai import OpenAI
import tiktoken
import os

from .medspaner_bridge import run_medspaner_question
from .utils_env import get_openai_api_key


# CONFIG

api_key = get_openai_api_key()
client = OpenAI(api_key=api_key)

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 350
TOP_K = 5

INDEX_PATH = "../vector_index/index.faiss"
META_PATH = "../vector_index/metadata.json"
MAP_PATH = "../vector_index/mapping.json"


# Utilidades

def clip_text(texto):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(texto)
    if len(tokens) <= MAX_TOKENS:
        return texto
    return enc.decode(tokens[:MAX_TOKENS])


def embed(text):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32")


def load_faiss():
    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(MAP_PATH, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    return index, metadata, mapping


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
        "forms": []
    }

    for ent in entities:

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
    meds_q = signals["meds"]
    diso_q = signals["diso"]
    forms_q = signals["forms"]

    if not meds_q and not diso_q and not forms_q:
        return candidates

    filtered = []

    for c in candidates:
        chunk_text = c["text"].lower()
        ents = c.get("entities", {})

        score = 0

        # coincidencia medicamento
        for m in meds_q:
            if m in chunk_text:
                score += 2

        # coincidencia patología
        for d in diso_q:
            if d in chunk_text:
                score += 1

        # coincidencia forma/vía
        for f in forms_q:
            if f in chunk_text:
                score += 1

        # agregar score de ajuste semántico
        c["_rerank_score"] = score

        filtered.append(c)

    # ordenar por score secundario + score vectorial
    filtered.sort(key=lambda x: (x["_rerank_score"], x["score"]), reverse=True)

    return filtered


# RETRIEVAL PRINCIPAL

def retrieve_chunks(query_text):

    # 1. Análisis médico con MEDSPANER (bridge)
    medspaner_output = run_medspaner_question(query_text)
    signals = extract_query_signals(medspaner_output)

    # 2. Embedding de la pregunta
    query_emb = embed(query_text)
    query_emb = query_emb.reshape(1, -1)
    faiss.normalize_L2(query_emb)

    # 3. Cargar FAISS y metadata
    index, metadata, mapping = load_faiss()

    # 4. Similaridad vectorial
    scores, idxs = index.search(query_emb, TOP_K)

    # 5. Construir lista de candidatos
    candidates = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        meta = dict(metadata[idx])
        meta["score"] = float(score)
        candidates.append(meta)

    # 6. Filtrado + reranking por entidades de MEDSPANER
    refined = filter_by_medical_signals(candidates, signals)

    return refined[:TOP_K], signals, medspaner_output
