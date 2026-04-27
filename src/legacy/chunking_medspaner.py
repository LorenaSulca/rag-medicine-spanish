import json
import re
import argparse
import os
from utils_env import get_data_dir

def normalize_newlines(text):
    return re.sub(r"\n{3,}", "\n\n", text)


def cargar_datos(text_path, json_path):
    with open(text_path, "r", encoding="utf-8") as f:
        texto = f.read()

    texto = normalize_newlines(texto)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "entities" in data:
        entidades = data["entities"]
    else:
        entidades = data 

    return texto, entidades



def generar_chunks(texto, chunk_size=800, overlap=200, ajuste_max=50):
    chunks = []
    n = len(texto)

    start = 0
    chunk_id = 0

    while start < n:
        end = min(start + chunk_size, n)

        ajuste = 0
        while end < n and ajuste < ajuste_max and texto[end] not in (" ", "\n"):
            end += 1
            ajuste += 1

        chunk_text = texto[start:end]

        chunks.append({
            "chunk_id": f"chunk_{chunk_id}",
            "text": chunk_text,
            "start": start,
            "end": end
        })

        chunk_id += 1
        next_start = end - overlap
        if next_start <= start:
            next_start = start + chunk_size

        start = next_start

    return chunks


def asignar_entidades(chunks, entidades):
    for chunk in chunks:
        chunk_entities = {}

        for ent in entidades:
            if chunk["start"] <= ent["start"] <= chunk["end"]:
                tipo = ent["entity_group"]
                palabra = ent["word"]

                if tipo not in chunk_entities:
                    chunk_entities[tipo] = []

                chunk_entities[tipo].append(palabra)

        chunk["entities"] = chunk_entities

    return chunks


def guardar_chunks(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunks guardados en: {output_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Chunking de prospectos con metadata de MEDSPANER")
    parser.add_argument("texto_file", help="Nombre del archivo TXT limpio")
    parser.add_argument("entidades_file", help="Nombre del archivo JSON de MEDSPANER")
    parser.add_argument("output_file", help="Nombre del archivo JSON a generar")
    args = parser.parse_args()

    DATA_DIR = os.path.dirname(os.getcwd()) + get_data_dir()

    path_texto = os.path.join(DATA_DIR, args.texto_file)
    path_entidades = os.path.join(DATA_DIR, args.entidades_file)
    path_output = os.path.join(DATA_DIR, args.output_file)

    if not os.path.exists(path_texto):
        raise FileNotFoundError(f"No existe el archivo de texto: {path_texto}")

    if not os.path.exists(path_entidades):
        raise FileNotFoundError(f"No existe el JSON de entidades: {path_entidades}")

    print(f"Archivo TXT: {path_texto}")
    print(f"Archivo entidades: {path_entidades}")
    print(f"Salida: {path_output}\n")

    texto, entidades = cargar_datos(path_texto, path_entidades)

    chunks = generar_chunks(texto)
    chunks = asignar_entidades(chunks, entidades)

    guardar_chunks(chunks, path_output)

    print("Chunking finalizado correctamente.")
