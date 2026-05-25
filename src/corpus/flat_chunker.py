import argparse
import json
import os
from typing import Optional


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def overlap_size(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def find_dominant_section(
    chunk_start: int,
    chunk_end: int,
    sections: list[dict],
) -> Optional[dict]:
    best_section = None
    best_overlap = 0

    for section in sections:
        section_start = int(section["start"])
        section_end = int(section["end"])

        ov = overlap_size(chunk_start, chunk_end, section_start, section_end)

        if ov > best_overlap:
            best_overlap = ov
            best_section = section

    return best_section


def chunk_full_text(
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunks.append({
                "local_start": start,
                "local_end": end,
                "text": chunk_text,
            })

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def build_flat_chunks(
    document_id: str,
    text: str,
    sections_data: Optional[dict] = None,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict]:
    sections = []

    if sections_data:
        sections = sections_data.get("sections", [])

    raw_chunks = chunk_full_text(
        text=text,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    output_chunks = []

    for idx, ch in enumerate(raw_chunks):
        start = ch["local_start"]
        end = ch["local_end"]

        section = find_dominant_section(start, end, sections) if sections else None

        output_chunks.append({
            "uid": f"{document_id}_flat_chunk_{idx:04d}",
            "document_id": document_id,
            "chunk_id": f"chunk_{idx}",
            "chunking_strategy": "flat",
            "section_id": section.get("section_id") if section else None,
            "section_name": section.get("section_name") if section else None,
            "section_number": section.get("section_number") if section else None,
            "start": start,
            "end": end,
            "text": ch["text"],
            "entities": {},
        })

    return output_chunks


def main():
    parser = argparse.ArgumentParser(
        description="Genera chunks planos desde el TXT completo, con asignación opcional de sección predominante."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del TXT limpio del prospecto.",
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="ID del documento, ej: paracetamol.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida JSON para chunks planos.",
    )

    parser.add_argument(
        "--sections",
        required=False,
        default=None,
        help="Ruta opcional al JSON de secciones para asignar section_name dominante.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Tamaño del chunk en caracteres.",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Solapamiento entre chunks en caracteres.",
    )

    args = parser.parse_args()

    text = load_text(args.input)

    sections_data = None
    if args.sections:
        sections_data = load_json(args.sections)

    chunks = build_flat_chunks(
        document_id=args.document_id,
        text=text,
        sections_data=sections_data,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    save_json(args.output, chunks)

    print(f"Chunks planos generados: {len(chunks)}")
    print(f"Archivo: {args.output}")

    with_sections = sum(1 for ch in chunks if ch.get("section_name"))
    print(f"Chunks con sección asignada: {with_sections}/{len(chunks)}")


if __name__ == "__main__":
    main()