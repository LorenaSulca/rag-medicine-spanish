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

BASE_OUTPUT_DIR = "../vector_index"


def get_index_paths(index_variant: str) -> dict:
    output_dir = os.path.join(BASE_OUTPUT_DIR, index_variant)

    return {
        "output_dir": output_dir,
        "index_path": os.path.join(output_dir, "index.faiss"),
        "meta_path": os.path.join(output_dir, "metadata.json"),
        "map_path": os.path.join(output_dir, "mapping.json"),
    }


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


def build_metadata(
    chunk: dict,
    documento_id: str,
    texto: str,
    index_variant: str,
) -> dict:
    chunk_uid = chunk.get("uid") or f"{documento_id}_{chunk['chunk_id']}"

    return {
        "uid": chunk_uid,
        "document_id": chunk.get("document_id", documento_id),
        "chunk_id": chunk["chunk_id"],

        "index_variant": index_variant,
        "chunking_strategy": chunk.get("chunking_strategy", index_variant),

        "section_id": chunk.get("section_id"),
        "section_name": chunk.get("section_name"),
        "section_number": chunk.get("section_number"),

        "entities": chunk.get("entities", {}),

        "start": chunk["start"],
        "end": chunk["end"],
        "text": texto,
    }


def indexar_faiss(
    chunks_json_path: str,
    documento_id: str,
    index_variant: str,
    reset_index: bool = False,
):
    paths = get_index_paths(index_variant)

    os.makedirs(paths["output_dir"], exist_ok=True)

    if reset_index:
        for path in [paths["index_path"], paths["meta_path"], paths["map_path"]]:
            if os.path.exists(path):
                os.remove(path)
        print(f"Índice anterior eliminado para variante: {index_variant}")

    chunks = load_chunks(chunks_json_path)

    print(f"Se encontraron {len(chunks)} chunks para indexar.")
    print(f"Variante de índice: {index_variant}")

    embeddings = []
    metadata_list = []

    for idx, chunk in enumerate(chunks):
        validate_chunk(chunk, idx)

        texto = clip_text_to_max_tokens(chunk["text"])
        metadata = build_metadata(
            chunk=chunk,
            documento_id=documento_id,
            texto=texto,
            index_variant=index_variant,
        )

        print(
            f"Embedding → {metadata['uid']} "
            f"| strategy={metadata.get('chunking_strategy')} "
            f"| section={metadata.get('section_name')}"
        )

        emb = generar_embedding(texto)

        embeddings.append(emb)
        metadata_list.append(metadata)

    if not embeddings:
        raise ValueError("No se generaron embeddings. Revisa el archivo de chunks.")

    matrix_new = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(matrix_new)

    if os.path.exists(paths["index_path"]):
        print("Cargando índice existente...")
        index = faiss.read_index(paths["index_path"])

        if index.d != matrix_new.shape[1]:
            raise ValueError("Dimensión de embedding incompatible con el índice existente.")

        index.add(matrix_new)
    else:
        print("Creando nuevo índice...")
        index = faiss.IndexFlatIP(matrix_new.shape[1])
        index.add(matrix_new)

    faiss.write_index(index, paths["index_path"])
    print(f"Índice actualizado guardado en {paths['index_path']}")

    if os.path.exists(paths["meta_path"]):
        with open(paths["meta_path"], "r", encoding="utf-8") as f:
            old_meta = json.load(f)
    else:
        old_meta = []

    merged_meta = old_meta + metadata_list

    with open(paths["meta_path"], "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2, ensure_ascii=False)

    print("Metadata actualizada.")

    if os.path.exists(paths["map_path"]):
        with open(paths["map_path"], "r", encoding="utf-8") as f:
            old_map = json.load(f)
    else:
        old_map = {}

    base_offset = len(old_map)
    new_map = {}

    for i, meta in enumerate(metadata_list):
        uid = meta["uid"]
        new_map[uid] = base_offset + i

    merged_map = {**old_map, **new_map}

    with open(paths["map_path"], "w", encoding="utf-8") as f:
        json.dump(merged_map, f, indent=2, ensure_ascii=False)

    print("Mapping actualizado.")
    print("\nIndexación FAISS completada.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Indexación FAISS para variantes de chunking."
    )

    parser.add_argument(
        "chunks_file",
        help="Nombre del archivo JSON dentro de data/chunks.",
    )

    parser.add_argument(
        "documento_id",
        nargs="?",
        default=None,
        help="ID del documento. Si no se indica, se usa el nombre del archivo sin .json.",
    )

    parser.add_argument(
        "--index-variant",
        choices=["flat", "sections"],
        required=True,
        help="Variante de índice a construir.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Elimina el índice anterior de esa variante antes de indexar.",
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
    print(f"Variante índice: {args.index_variant}")
    print(f"Reset index: {args.reset}\n")

    indexar_faiss(
        chunks_json_path=chunks_json_path,
        documento_id=documento_id,
        index_variant=args.index_variant,
        reset_index=args.reset,
    )

    print("Indexación FAISS finalizada.")