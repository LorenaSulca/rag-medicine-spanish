import argparse
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from dataset.qa_generator import (
    generate_qa_for_section,
    generate_out_of_context_qa,
    assign_ids,
)


def load_sections(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Genera dataset QA clasificado para evaluación RAG."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Archivo JSON con document_id y secciones.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida del dataset QA.",
    )

    parser.add_argument(
        "--per-section",
        type=int,
        default=5,
        help="Cantidad de QA por sección.",
    )

    parser.add_argument(
        "--out-of-context",
        type=int,
        default=5,
        help="Cantidad de QA fuera de contexto.",
    )

    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Modelo usado para generar QA.",
    )

    args = parser.parse_args()

    load_dotenv()
    client = OpenAI()

    data = load_sections(args.input)
    document_id = data["document_id"]
    sections = data["sections"]

    all_items = []

    for section in sections:
        section_name = section["section_name"]
        section_text = section["section_text"]

        print(f"Generando QA para sección: {section_name}")

        items = generate_qa_for_section(
            client=client,
            document_id=document_id,
            section_name=section_name,
            section_text=section_text,
            n=args.per_section,
            model=args.model,
        )

        all_items.extend(items)

    known_sections = [s["section_name"] for s in sections]

    if args.out_of_context > 0:
        print("Generando preguntas fuera de contexto...")

        ooc_items = generate_out_of_context_qa(
            client=client,
            document_id=document_id,
            known_sections=known_sections,
            n=args.out_of_context,
            model=args.model,
        )

        all_items.extend(ooc_items)

    all_items = assign_ids(all_items, prefix=document_id)

    save_json(args.output, all_items)

    print(f"Dataset generado: {args.output}")
    print(f"Total QA: {len(all_items)}")


if __name__ == "__main__":
    main()