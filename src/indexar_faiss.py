import json
import numpy as np
import faiss
import tiktoken
from openai import OpenAI
from utils_env import get_openai_api_key
from utils_env import get_data_dir
import os
import argparse

api_key = get_openai_api_key()
client = OpenAI(api_key=api_key)

EMBEDDING_MODEL = "text-embedding-3-small"
MAX_TOKENS = 800

OUTPUT_DIR = "../vector_index"
INDEX_PATH = f"{OUTPUT_DIR}/index.faiss"
META_PATH = f"{OUTPUT_DIR}/metadata.json"
MAP_PATH = f"{OUTPUT_DIR}/mapping.json"


# Token clipping

def clip_text_to_max_tokens(texto):
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(texto)

    if len(tokens) <= MAX_TOKENS:
        return texto

    tokens = tokens[:MAX_TOKENS]
    return enc.decode(tokens)


# Generate embedding

def generar_embedding(texto):
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texto
    )
    return np.array(resp.data[0].embedding, dtype="float32")


# Indexing pipeline

def indexar_faiss(chunks_json_path, documento_id):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Cargar chunks
    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Se encontraron {len(chunks)} chunks para indexar.")

    embeddings = []
    metadata_list = []
    id_map = {}

    # Producir embeddings
    for idx, chunk in enumerate(chunks):
        chunk_uid = f"{documento_id}_{chunk['chunk_id']}"
        texto = clip_text_to_max_tokens(chunk["text"])

        print(f"Embedding → {chunk_uid}")

        emb = generar_embedding(texto)
        embeddings.append(emb)

        metadata_list.append({
            "uid": chunk_uid,
            "document_id": documento_id,
            "chunk_id": chunk["chunk_id"],
            "entities": chunk.get("entities", {}),
            "start": chunk["start"],
            "end": chunk["end"],
            "text": texto
        })

    # Convertir a matriz
    matrix_new = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(matrix_new)

    # Cargar índice anterior si existe 
    if os.path.exists(INDEX_PATH):
        print("Cargando índice existente…")
        index = faiss.read_index(INDEX_PATH)

        # Validar dimensión
        if index.d != matrix_new.shape[1]:
            raise ValueError("Dimensión de embedding incompatible con el índice existente.")

        index.add(matrix_new)
    else:
        print("Creando nuevo indice")
        index = faiss.IndexFlatIP(matrix_new.shape[1])
        index.add(matrix_new)

    # Guardar índice actualizado
    faiss.write_index(index, INDEX_PATH)
    print(f"Indice actualizado guardado en {INDEX_PATH}")

    # Metadata
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            old_meta = json.load(f)
    else:
        old_meta = []

    merged_meta = old_meta + metadata_list

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_meta, f, indent=2, ensure_ascii=False)
    print("Metadata actualizada.")

    # Mapping
    if os.path.exists(MAP_PATH):
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            old_map = json.load(f)
    else:
        old_map = {}

    # Indice FAISS simplemente agrega vectores al final del index
    base_offset = len(old_map)

    for i, meta in enumerate(metadata_list):
        uid = meta["uid"]
        id_map[uid] = base_offset + i

    merged_map = {**old_map, **id_map}

    with open(MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(merged_map, f, indent=2, ensure_ascii=False)
    print("Mapping actualizado.")

    print("\n✓ Indexación incremental completada.\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexación FAISS para chunks de medicamentos")
    parser.add_argument("chunks_file", help="Nombre del archivo JSON con chunks, ej: paracetamol_chunks.json")
    parser.add_argument(
        "documento_id",
        nargs="?",
        default=None,
        help="ID del documento (opcional). Si no se indica, se usa el nombre del archivo sin .json"
    )
    
    args = parser.parse_args()

    DATA_DIR = get_data_dir()

    chunks_json_path = os.path.join(DATA_DIR, args.chunks_file)

    if not os.path.exists(chunks_json_path):
        raise FileNotFoundError(f"No existe el archivo de chunks: {chunks_json_path}")

    # Si no se pasa documento_id, usar nombre del archivo sin .json
    documento_id = args.documento_id
    if documento_id is None:
        documento_id = os.path.splitext(args.chunks_file)[0]

    print("\n=== Indexando FAISS ===")
    print(f"Archivo chunks: {chunks_json_path}")
    print(f"Documento ID: {documento_id}\n")

    indexar_faiss(
        chunks_json_path=chunks_json_path,
        documento_id=documento_id
    )

    print("\nIndexación FAISS finalizada.\n")
