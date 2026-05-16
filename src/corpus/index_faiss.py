import argparse
import json
import os

import faiss
import numpy as np
import tiktoken
from openai import OpenAI

from retrieval.utils_env import get_openai_api_key, get_data_dir


api_key = get_openai_api_key()
client = OpenAI(api_key=api_key)

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 800

OUTPUT_DIR = "../vector_index"
INDEX_PATH = os.path.join(OUTPUT_DIR, "index.faiss")
META_PATH = os.path.join(OUTPUT_DIR, "metadata.json")
MAP_PATH = os.path.join(OUTPUT_DIR, "mapping.json")


def clip_text_to_max_tokens(texto: str) -> str:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(texto)

    if len(tokens) <= MAX_TOKENS:
        return texto

    return enc.decode(tokens[:MAX_TOKENS])


def generar_embedding(texto: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texto,
    )
    return np.array(resp.data[0].embedding, dtype="float32")


def load_chunks(chunks_json_path: str) -> list[dict]:
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("El archivo de chunks debe contener una lista JSON.")

    return chunks


def validate_chunk(chunk: dict, idx: int):
    required = ["chunk_id", "text", "start", "end"]

    for field in required:
        if field not in chunk:
            raise ValueError(f"Chunk #{idx} no tiene el campo requerido: {field}")

    if not chunk["text"].strip():
        raise ValueError(f"Chunk #{idx} tiene texto vacío.")


def build_metadata(chunk: dict, documento_id: str, texto: str) -> dict:
    chunk_uid = chunk.get("uid") or f"{documento_id}_{chunk['chunk_id']}"

    return {
        "uid": chunk_uid,
        "document_id": chunk.get("document_id", documento_id),
        "chunk_id": chunk["chunk_id"],

        # Metadata estructural por sección
        "section_id": chunk.get("section_id"),
        "section_name": chunk.get("section_name"),
        "section_number": chunk.get("section_number"),

        # Metadata biomédica
        "entities": chunk.get("entities", {}),

        # Posición documental
        "start": chunk["start"],
        "end": chunk["end"],

        # Texto finalmente indexado
        "text": texto,
    }


def indexar_faiss(
    chunks_json_path: str,
    documento_id: str,
    reset_index: bool = False,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if reset_index:
        for path in [INDEX_PATH, META_PATH, MAP_PATH]:
            if os.path.exists(path):
                os.remove(path)
        print("Índice anterior eliminado.")

    chunks = load_chunks(chunks_json_path)

    print(f"Se encontraron {len(chunks)} chunks para indexar.")

    embeddings = []
    metadata_list = []

    for idx, chunk in enumerate(chunks):
        validate_chunk(chunk, idx)

        texto = clip_text_to_max_tokens(chunk["text"])
        metadata = build_metadata(chunk, documento_id, texto)

        print(
            f"Embedding → {metadata['uid']} "
            f"| sección={metadata.get('section_name')}"
        )

        emb = generar_embedding(texto)

        embeddings.append(emb)
        metadata_list.append(metadata)

    if not embeddings:
        raise ValueError("No se generaron embeddings. Revisa el archivo de chunks.")

    matrix_new = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(matrix_new)

    if os.path.exists(INDEX_PATH):
        print("Cargando índice existente...")
        index = faiss.read_index(INDEX_PATH)

        if index.d != matrix_new.shape[1]:
            raise ValueError("Dimensión de embedding incompatible con el índice existente.")

        index.add(matrix_new)
    else:
        print("Creando nuevo índice...")
        index = faiss.IndexFlatIP(matrix_new.shape[1])
        index.add(matrix_new)

    faiss.write_index(index, INDEX_PATH)
    print(f"Índice actualizado guardado en {INDEX_PATH}")

    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            old_meta = json.load(f)
    else:
        old_meta = []

    merged_meta = old_meta + metadata_list

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2, ensure_ascii=False)

    print("Metadata actualizada.")

    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            old_map = json.load(f)
    else:
        old_map = {}

    base_offset = len(old_map)
    new_map = {}

    for i, meta in enumerate(metadata_list):
        uid = meta["uid"]
        new_map[uid] = base_offset + i

    merged_map = {**old_map, **new_map}

    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_map, f, indent=2, ensure_ascii=False)

    print("Mapping actualizado.")
    print("\nIndexación FAISS completada.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indexación FAISS para chunks seccionados de medicamentos."
    )

    parser.add_argument(
        "chunks_file",
        help="Nombre del archivo JSON dentro de data/chunks, ej: paracetamol_chunks_sectioned.json",
    )

    parser.add_argument(
        "documento_id",
        nargs="?",
        default=None,
        help="ID del documento. Si no se indica, se usa el nombre del archivo sin .json",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina el índice anterior antes de indexar.",
    )

    args = parser.parse_args()

    data_dir = os.path.dirname(os.getcwd()) + get_data_dir()
    chunks_json_path = os.path.join(data_dir, "chunks", args.chunks_file)

    if not os.path.exists(chunks_json_path):
        raise FileNotFoundError(f"No existe el archivo de chunks: {chunks_json_path}")

    documento_id = args.documento_id

    if documento_id is None:
        documento_id = os.path.splitext(args.chunks_file)[0]

    print("\n=== Indexando FAISS ===")
    print(f"Archivo chunks: {chunks_json_path}")
    print(f"Documento ID: {documento_id}")
    print(f"Reset index: {args.reset}\n")

    indexar_faiss(
        chunks_json_path=chunks_json_path,
        documento_id=documento_id,
        reset_index=args.reset,
    )

    print("Indexación FAISS finalizada.")