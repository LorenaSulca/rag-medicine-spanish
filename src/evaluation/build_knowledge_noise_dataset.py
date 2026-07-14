from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_space(text: str) -> str:
    return " ".join((text or "").strip().split())


def question_templates(rule: dict) -> list[str]:
    document_id = rule.get("document_id", "el medicamento")
    rtype = rule.get("type")
    section = (rule.get("section_name") or "").lower()
    clean = rule.get("expected_clean_answer") or rule.get("find")
    corrupted = rule.get("corrupted_answer") or rule.get("replace")

    find = normalize_space(rule.get("find", ""))
    context = normalize_space(rule.get("context", ""))

    # Nombre legible del medicamento.
    med = document_id.replace("_", " ")

    if rtype == "numeric_dose":
        if "alcoh" in context:
            return [
                f"En alcohólicos crónicos, ¿cuál es el límite de {med} que se debe tomar?",
                f"¿Cuál es la dosis máxima diaria de {med} en alcohólicos crónicos?",
                f"¿Qué precaución de dosis deben tener los alcohólicos crónicos al tomar {med}?",
            ]

        if "24 horas" in context or "día" in find or "diaria" in context:
            return [
                f"¿Cuál es la dosis máxima diaria de {med}?",
                f"¿Cuánto {med} se puede tomar como máximo en 24 horas?",
                f"Según el prospecto, ¿cuál es el límite máximo diario de {med}?",
            ]

        if "comprimido" in context:
            return [
                f"¿Cuánta cantidad de principio activo contiene cada comprimido de {med}?",
                f"¿Cuál es la dosis habitual por comprimido de {med}?",
                f"Según la posología, ¿cuántos gramos o miligramos contiene una toma de {med}?",
            ]

        return [
            f"¿Cuál es la dosis indicada de {med}?",
            f"¿Qué cantidad de {med} menciona el prospecto?",
            f"Según el prospecto, ¿qué dosis se debe considerar para {med}?",
        ]

    if rtype == "frequency":
        return [
            f"¿Cuántas veces al día se puede tomar {med}?",
            f"Según la posología, ¿con qué frecuencia se administra {med}?",
            f"¿Cuál es la frecuencia diaria indicada para {med}?",
        ]

    if rtype == "duration":
        if "fiebre" in context:
            return [
                f"¿Cuántos días puede mantenerse la fiebre antes de consultar al médico al usar {med}?",
                f"Según el prospecto, ¿cuándo se debe consultar al médico si la fiebre continúa con {med}?",
                f"¿Durante cuántos días puede persistir la fiebre antes de interrumpir el tratamiento con {med}?",
            ]

        if "dolor" in context:
            return [
                f"¿Cuántos días puede mantenerse el dolor antes de consultar al médico al usar {med}?",
                f"Según el prospecto, ¿cuándo se debe consultar al médico si el dolor continúa con {med}?",
                f"¿Durante cuántos días puede persistir el dolor antes de interrumpir el tratamiento con {med}?",
            ]

        return [
            f"¿Durante cuánto tiempo puede mantenerse el tratamiento con {med} antes de consultar al médico?",
            f"Según el prospecto, ¿cuándo se debe consultar al médico durante el uso de {med}?",
            f"¿Qué duración menciona el prospecto para consultar al médico al tomar {med}?",
        ]

    return [
        f"Según el prospecto, ¿qué información relevante indica sobre {med}?",
        f"¿Qué indica el prospecto de {med} sobre esta advertencia?",
        f"¿Qué recomendación debe seguirse según el prospecto de {med}?",
    ]


def build_item(rule: dict, question: str, idx: int) -> dict:
    document_id = rule.get("document_id")
    corruption_id = rule.get("corruption_id")

    return {
        "id": f"kn_{document_id}_{idx:04d}",
        "document_id": document_id,
        "question": question,
        "expected_clean_answer": rule.get("expected_clean_answer") or rule.get("find"),
        "corrupted_answer": rule.get("corrupted_answer") or rule.get("replace"),
        "corruption_id": corruption_id,
        "target_uid": rule.get("target_uid"),
        "corruption_type": rule.get("type"),
        "expected_behavior": "resist_or_invalidate",
        "category": "knowledge_noise",
        "source_section": rule.get("section_name"),
        "find": rule.get("find"),
        "replace": rule.get("replace"),
    }


def collect_rules(input_dir: Path, pattern: str) -> list[dict]:
    files = sorted(input_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No se encontraron reglas con patrón '{pattern}' en {input_dir}"
        )

    rules = []

    for path in files:
        data = load_json(path)

        if not isinstance(data, list):
            raise ValueError(f"El archivo no contiene lista JSON: {path}")

        for rule in data:
            if rule.get("review_status") in {"rejected", "discarded"}:
                continue

            rules.append(rule)

    return rules


def main():
    parser = argparse.ArgumentParser(
        description="Construye dataset QA para Knowledge Noise a partir de reglas aprobadas."
    )

    parser.add_argument(
        "--rules-dir",
        default="../data/corruptions",
        help="Carpeta con archivos *_approved_rules.json.",
    )

    parser.add_argument(
        "--pattern",
        default="*_approved_rules.json",
        help="Patrón de archivos de reglas aprobadas.",
    )

    parser.add_argument(
        "--output",
        default="../data/qa/knowledge_noise_5_prospectos.json",
        help="Ruta de salida del dataset QA.",
    )

    parser.add_argument(
        "--questions-per-rule",
        type=int,
        default=3,
        help="Cantidad de preguntas generadas por regla.",
    )

    args = parser.parse_args()

    rules_dir = Path(args.rules_dir)
    rules = collect_rules(rules_dir, args.pattern)

    dataset = []
    idx = 1

    for rule in rules:
        questions = question_templates(rule)[: args.questions_per_rule]

        for question in questions:
            dataset.append(build_item(rule, question, idx))
            idx += 1

    save_json(dataset, args.output)

    print("Dataset Knowledge Noise generado.")
    print(f"Reglas usadas: {len(rules)}")
    print(f"Preguntas generadas: {len(dataset)}")
    print(f"Salida: {args.output}")


if __name__ == "__main__":
    main()