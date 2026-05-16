import argparse
import json
import os
import re


SECTION_PATTERNS = [
    r"1\.\s*Qué es.*?(?=\n2\.|\Z)",
    r"2\.\s*Qué necesita saber.*?(?=\n3\.|\Z)",
    r"3\.\s*Cómo tomar.*?(?=\n4\.|\Z)",
    r"4\.\s*Posibles efectos adversos.*?(?=\n5\.|\Z)",
    r"5\.\s*Conservación.*?(?=\n6\.|\Z)",
    r"6\.\s*Contenido del envase.*?(?=\Z)",
]


def clean_section_name(raw: str) -> str:
    raw = " ".join(raw.split())
    raw = raw.strip()

    if len(raw) > 120:
        raw = raw[:120]

    return raw


def split_sections(text: str) -> list[dict]:
    sections = []

    for pattern in SECTION_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

        if not match:
            continue

        section_text = match.group(0).strip()

        first_line = section_text.split("\n", 1)[0]
        section_name = clean_section_name(first_line)

        sections.append({
            "section_name": section_name,
            "section_text": section_text
        })

    return sections


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def save_json(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Extrae secciones estructuradas desde un TXT limpio de prospecto."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del TXT limpio extraído del prospecto."
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="Identificador del documento, ej: paracetamol."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Ruta JSON de salida."
    )

    args = parser.parse_args()

    text = load_text(args.input)
    sections = split_sections(text)

    if not sections:
        raise ValueError(
            "No se detectaron secciones. Revisa si el prospecto usa numeración 1., 2., 3., etc."
        )

    output = {
        "document_id": args.document_id,
        "sections": sections
    }

    save_json(args.output, output)

    print(f"Secciones extraídas: {len(sections)}")
    print(f"Archivo generado: {args.output}")

    for s in sections:
        print("-", s["section_name"])


if __name__ == "__main__":
    main()