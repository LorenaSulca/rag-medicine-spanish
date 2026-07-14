from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DOSE_PATTERNS = [
    {
        "type": "numeric_dose",
        "pattern": re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(mg|g)\s*/\s*día\b", re.IGNORECASE),
        "transform": "double_value",
    },
    {
        "type": "numeric_dose",
        "pattern": re.compile(r"\b(\d+(?:[.,]\d+)?)\s*(mg|g)\b", re.IGNORECASE),
        "transform": "double_value",
    },
]

FREQUENCY_PATTERNS = [
    {
        "type": "frequency",
        "pattern": re.compile(r"\bcada\s+(\d+)\s+horas\b", re.IGNORECASE),
        "transform": "halve_hours",
    },
    {
        "type": "frequency",
        "pattern": re.compile(r"\b(\d+)\s*-\s*(\d+)\s+veces\s+al\s+día\b", re.IGNORECASE),
        "transform": "double_range",
    },
    {
        "type": "frequency",
        "pattern": re.compile(r"\b(\d+)\s+veces\s+al\s+día\b", re.IGNORECASE),
        "transform": "double_value",
    },
]

CONTRAINDICATION_PATTERNS = [
    {
        "type": "contraindication_negation",
        "pattern": re.compile(r"\bNo tome\b", re.IGNORECASE),
        "replace": "Tome",
    },
    {
        "type": "contraindication_negation",
        "pattern": re.compile(r"\bNo debe\b", re.IGNORECASE),
        "replace": "Puede",
    },
    {
        "type": "contraindication_negation",
        "pattern": re.compile(r"\bno tomar\b", re.IGNORECASE),
        "replace": "tomar",
    },
]

DURATION_PATTERNS = [
    {
        "type": "duration",
        "pattern": re.compile(r"\bmás de\s+(\d+)\s+días\b", re.IGNORECASE),
        "transform": "double_value",
    },
]


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_decimal(value: str) -> float:
    return float(value.replace(",", "."))


def format_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", ",")


def double_numeric_match(match: re.Match) -> str:
    value = normalize_decimal(match.group(1))
    unit = match.group(2)

    new_value = value * 2

    return f"{format_number(new_value)} {unit}"


def double_numeric_per_day_match(match: re.Match) -> str:
    value = normalize_decimal(match.group(1))
    unit = match.group(2)

    new_value = value * 2

    return f"{format_number(new_value)} {unit}/día"


def halve_hours_match(match: re.Match) -> str:
    hours = int(match.group(1))

    if hours <= 1:
        return match.group(0)

    new_hours = max(1, hours // 2)

    return f"cada {new_hours} horas"


def double_range_match(match: re.Match) -> str:
    start = int(match.group(1))
    end = int(match.group(2))

    return f"{start * 2}-{end * 2} veces al día"


def double_single_frequency_match(match: re.Match) -> str:
    value = int(match.group(1))
    return f"{value * 2} veces al día"


def double_duration_match(match: re.Match) -> str:
    value = int(match.group(1))
    return f"más de {value * 2} días"


def get_context_window(text: str, start: int, end: int, window: int = 120) -> str:
    left = max(0, start - window)
    right = min(len(text), end + window)

    return " ".join(text[left:right].split())


def make_corruption_id(document_id: str, chunk_id: str, corruption_type: str, index: int) -> str:
    clean_doc = re.sub(r"[^a-zA-Z0-9_]+", "_", document_id.lower())
    clean_chunk = re.sub(r"[^a-zA-Z0-9_]+", "_", chunk_id.lower())

    return f"{clean_doc}_{clean_chunk}_{corruption_type}_{index:03d}"


def suggest_from_regex_patterns(
    chunk: dict,
    document_id: str,
    patterns: list[dict],
    start_index: int,
) -> list[dict]:
    text = chunk.get("text", "")
    uid = chunk.get("uid")
    chunk_id = chunk.get("chunk_id", "chunk")
    suggestions = []

    counter = start_index

    for spec in patterns:
        pattern = spec["pattern"]
        corruption_type = spec["type"]

        for match in pattern.finditer(text):
            original = match.group(0)

            if spec.get("replace"):
                corrupted = spec["replace"]
            elif spec.get("transform") == "double_value" and corruption_type == "numeric_dose":
                # Mantiene soporte para mg/g sin /día.
                corrupted = double_numeric_match(match)
            elif spec.get("transform") == "double_value" and corruption_type == "duration":
                corrupted = double_duration_match(match)
            elif spec.get("transform") == "double_value" and corruption_type == "frequency":
                corrupted = double_single_frequency_match(match)
            elif spec.get("transform") == "double_value":
                corrupted = original
            elif spec.get("transform") == "halve_hours":
                corrupted = halve_hours_match(match)
            elif spec.get("transform") == "double_range":
                corrupted = double_range_match(match)
            elif spec.get("transform") == "double_numeric_per_day":
                corrupted = double_numeric_per_day_match(match)
            else:
                corrupted = original

            # Caso especial para dosis con /día.
            if corruption_type == "numeric_dose" and "/día" in original.lower():
                corrupted = double_numeric_per_day_match(match)

            if corrupted == original:
                continue

            suggestions.append({
                "corruption_id": make_corruption_id(
                    document_id=document_id,
                    chunk_id=chunk_id,
                    corruption_type=corruption_type,
                    index=counter,
                ),
                "document_id": document_id,
                "target_uid": uid,
                "chunk_id": chunk_id,
                "section_name": chunk.get("section_name"),
                "section_number": chunk.get("section_number"),
                "type": corruption_type,
                "find": original,
                "replace": corrupted,
                "expected_clean_answer": original,
                "corrupted_answer": corrupted,
                "context": get_context_window(text, match.start(), match.end()),
                "review_status": "pending",
                "auto_generated": True,
            })

            counter += 1

    return suggestions


def suggest_rules_for_chunks(chunks: list[dict], document_id: str) -> list[dict]:
    all_patterns = (
        DOSE_PATTERNS
        + FREQUENCY_PATTERNS
        + CONTRAINDICATION_PATTERNS
        + DURATION_PATTERNS
    )

    suggestions = []
    counter = 1

    for chunk in chunks:
        text = chunk.get("text", "")

        if not text.strip():
            continue

        # Evitar chunks que son solo títulos cortos.
        if len(text.split()) < 8:
            continue

        chunk_suggestions = suggest_from_regex_patterns(
            chunk=chunk,
            document_id=document_id,
            patterns=all_patterns,
            start_index=counter,
        )

        suggestions.extend(chunk_suggestions)
        counter += len(chunk_suggestions)

    return suggestions


def main():
    parser = argparse.ArgumentParser(
        description="Sugiere reglas de corrupción para Knowledge Noise sin modificar chunks."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta del JSON de chunks limpio.",
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="ID del documento/prospecto. Ej: paracetamol.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Ruta de salida del JSON de reglas sugeridas.",
    )

    args = parser.parse_args()

    chunks = load_json(args.input)

    if not isinstance(chunks, list):
        raise ValueError("El archivo de chunks debe contener una lista JSON.")

    suggestions = suggest_rules_for_chunks(
        chunks=chunks,
        document_id=args.document_id,
    )

    save_json(suggestions, args.output)

    print("Sugerencias generadas.")
    print(f"Documento: {args.document_id}")
    print(f"Chunks: {len(chunks)}")
    print(f"Sugerencias: {len(suggestions)}")
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()