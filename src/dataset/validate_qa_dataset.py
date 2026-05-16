import argparse
import json
from collections import Counter


REQUIRED_FIELDS = [
    "id",
    "document_id",
    "category",
    "difficulty",
    "question",
    "ground_truth",
    "source_sections",
    "expected_behavior",
    "requires_numeric_grounding",
    "requires_multi_hop",
    "notes",
]

VALID_CATEGORIES = {
    "indications",
    "dosage",
    "contraindications",
    "interactions",
    "pregnancy_lactation",
    "adverse_effects",
    "overdose",
    "administration",
    "storage",
    "warnings",
    "out_of_context",
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_BEHAVIORS = {"answered", "partially answered", "abstained"}


def load_dataset(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("El dataset debe ser una lista JSON.")

    return data


def validate_item(item: dict, idx: int) -> list[str]:
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in item:
            errors.append(f"Item #{idx}: falta campo '{field}'")

    if errors:
        return errors

    if item["category"] not in VALID_CATEGORIES:
        errors.append(f"Item #{idx}: categoría inválida: {item['category']}")

    if item["difficulty"] not in VALID_DIFFICULTIES:
        errors.append(f"Item #{idx}: dificultad inválida: {item['difficulty']}")

    if item["expected_behavior"] not in VALID_BEHAVIORS:
        errors.append(f"Item #{idx}: expected_behavior inválido: {item['expected_behavior']}")

    if not isinstance(item["source_sections"], list):
        errors.append(f"Item #{idx}: source_sections debe ser una lista")

    if not isinstance(item["requires_numeric_grounding"], bool):
        errors.append(f"Item #{idx}: requires_numeric_grounding debe ser booleano")

    if not isinstance(item["requires_multi_hop"], bool):
        errors.append(f"Item #{idx}: requires_multi_hop debe ser booleano")

    if not item["question"].strip():
        errors.append(f"Item #{idx}: question vacío")

    if not item["ground_truth"].strip():
        errors.append(f"Item #{idx}: ground_truth vacío")

    if item["expected_behavior"] == "abstained":
        if item["source_sections"]:
            errors.append(f"Item #{idx}: abstained no debería tener source_sections")

    if item["category"] == "out_of_context":
        if item["expected_behavior"] != "abstained":
            errors.append(f"Item #{idx}: out_of_context debe tener expected_behavior='abstained'")

    return errors


def summarize(data: list[dict]):
    print("\n===== RESUMEN DATASET QA =====")
    print(f"Total QA: {len(data)}")

    print("\nPor documento:")
    for k, v in Counter(item["document_id"] for item in data).items():
        print(f"- {k}: {v}")

    print("\nPor categoría:")
    for k, v in Counter(item["category"] for item in data).items():
        print(f"- {k}: {v}")

    print("\nPor dificultad:")
    for k, v in Counter(item["difficulty"] for item in data).items():
        print(f"- {k}: {v}")

    print("\nPor comportamiento esperado:")
    for k, v in Counter(item["expected_behavior"] for item in data).items():
        print(f"- {k}: {v}")

    numeric_count = sum(1 for item in data if item["requires_numeric_grounding"])
    multihop_count = sum(1 for item in data if item["requires_multi_hop"])

    print(f"\nRequieren grounding numérico: {numeric_count}")
    print(f"Requieren multi-hop: {multihop_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Valida estructura y distribución del dataset QA clasificado."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Ruta al dataset QA JSON."
    )

    args = parser.parse_args()

    data = load_dataset(args.input)

    errors = []

    seen_ids = set()

    for idx, item in enumerate(data, start=1):
        item_errors = validate_item(item, idx)
        errors.extend(item_errors)

        item_id = item.get("id")
        if item_id in seen_ids:
            errors.append(f"Item #{idx}: id duplicado: {item_id}")
        seen_ids.add(item_id)

    summarize(data)

    print("\n===== VALIDACIÓN =====")
    if errors:
        print(f"Errores encontrados: {len(errors)}")
        for e in errors[:50]:
            print("-", e)

        if len(errors) > 50:
            print(f"... y {len(errors) - 50} errores más")

        raise SystemExit(1)

    print("Dataset válido.")


if __name__ == "__main__":
    main()