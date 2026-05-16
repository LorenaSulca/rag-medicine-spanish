import argparse
import json
import os
from collections import defaultdict


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_entities(raw_medspaner: dict | list) -> list[dict]:
    """
    Soporta dos formatos:
    1. {"entities": [...]}
    2. [...]
    """
    if isinstance(raw_medspaner, dict) and "entities" in raw_medspaner:
        entities = raw_medspaner["entities"]
    else:
        entities = raw_medspaner

    if not isinstance(entities, list):
        return []

    normalized = []

    for ent in entities:
        if not isinstance(ent, dict):
            continue

        start = ent.get("start")
        end = ent.get("end")

        if start is None or end is None:
            # Si MEDSPANER no devuelve offsets, no se puede alinear por posición.
            continue

        normalized.append({
            "text": ent.get("word") or ent.get("text") or ent.get("entity") or "",
            "label": ent.get("entity_group") or ent.get("label") or ent.get("type") or "",
            "start": int(start),
            "end": int(end),
            "raw": ent,
        })

    return normalized


def group_entities_for_chunk(entities: list[dict], chunk_start: int, chunk_end: int) -> dict:
    grouped = defaultdict(list)

    for ent in entities:
        ent_start = ent["start"]
        ent_end = ent["end"]

        overlaps = ent_start < chunk_end and ent_end > chunk_start

        if overlaps:
            label = ent["label"] or "UNKNOWN"
            grouped[label].append({
                "text": ent["text"],
                "start": ent_start,
                "end": ent_end,
            })

    return dict(grouped)


def align_entities_to_chunks(chunks: list[dict], entities: list[dict]) -> list[dict]:
    enriched = []

    for chunk in chunks:
        chunk_start = int(chunk["start"])
        chunk_end = int(chunk["end"])

        new_chunk = dict(chunk)
        new_chunk["entities"] = group_entities_for_chunk(
            entities,
            chunk_start,
            chunk_end,
        )

        enriched.append(new_chunk)

    return enriched


def main():
    parser = argparse.ArgumentParser(
        description="Alinea entidades MEDSPANER a chunks por offsets."
    )

    parser.add_argument("--chunks", required=True)
    parser.add_argument("--entities", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    chunks = load_json(args.chunks)
    raw_entities = load_json(args.entities)

    entities = normalize_entities(raw_entities)

    if not entities:
        print("Advertencia: no se encontraron entidades con offsets start/end.")
        print("Se guardarán chunks con entities vacío.")

    enriched = align_entities_to_chunks(chunks, entities)

    save_json(args.output, enriched)

    print(f"Chunks enriquecidos: {len(enriched)}")
    print(f"Entidades normalizadas: {len(entities)}")
    print(f"Archivo generado: {args.output}")


if __name__ == "__main__":
    main()