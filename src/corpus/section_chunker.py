import argparse
import json
import os


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def chunk_text_by_chars(
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
                "text": chunk_text,
                "local_start": start,
                "local_end": end,
            })

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def build_sectioned_chunks(
    sections_data: dict,
    chunk_size: int = 1200,
    overlap: int = 200,
):
    document_id = sections_data["document_id"]
    output_chunks = []
    counter = 0

    for section in sections_data["sections"]:
        section_chunks = chunk_text_by_chars(
            section["section_text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for ch in section_chunks:
            output_chunks.append({
                "uid": f"{document_id}_chunk_{counter:04d}",
                "document_id": document_id,
                "chunk_id": f"chunk_{counter}",
                "section_id": section["section_id"],
                "section_name": section["section_name"],
                "section_number": section["section_number"],
                "start": section["start"] + ch["local_start"],
                "end": section["start"] + ch["local_end"],
                "text": ch["text"],
                "entities": {},
            })
            counter += 1

    return output_chunks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    sections_data = load_json(args.input)

    chunks = build_sectioned_chunks(
        sections_data,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )

    save_json(args.output, chunks)

    print(f"Chunks generados: {len(chunks)}")
    print(f"Archivo: {args.output}")


if __name__ == "__main__":
    main()