from __future__ import annotations

import argparse
import json
from pathlib import Path


PREFERRED_TYPES = {"numeric_dose", "frequency", "duration"}
REJECT_TYPES = {"contraindication_negation"}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_duplicate(rule, seen):
    key = (
        rule.get("target_uid"),
        rule.get("type"),
        rule.get("find"),
        rule.get("replace"),
    )
    if key in seen:
        return True
    seen.add(key)
    return False


def score_rule(rule):
    score = 0
    text = (rule.get("context") or "").lower()
    rtype = rule.get("type")

    if rtype in PREFERRED_TYPES:
        score += 10

    if rtype in REJECT_TYPES:
        score -= 100

    if "dosis" in text or "posología" in text:
        score += 5

    if "no se tomarán más" in text or "no tomar más" in text:
        score += 5

    if "cada" in text and "horas" in text:
        score += 4

    if "veces al día" in text:
        score += 4

    if "más de" in text and "días" in text:
        score += 3

    if "composición" in text or "contenido del envase" in text:
        score -= 6

    if rule.get("find") in {"1 g", "2 g", "4 g"} and "/" not in rule.get("find", ""):
        score -= 2

    return score


def filter_rules(rules, max_per_document=6):
    seen = set()
    filtered = []

    for rule in rules:
        if rule.get("type") in REJECT_TYPES:
            continue

        if rule.get("type") not in PREFERRED_TYPES:
            continue

        if is_duplicate(rule, seen):
            continue

        rule = dict(rule)
        rule["_score"] = score_rule(rule)
        filtered.append(rule)

    filtered.sort(key=lambda r: r["_score"], reverse=True)

    return filtered[:max_per_document]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max", type=int, default=6)
    args = parser.parse_args()

    rules = load_json(args.input)
    selected = filter_rules(rules, max_per_document=args.max)

    save_json(selected, args.output)

    print(f"Entrada: {len(rules)} reglas")
    print(f"Seleccionadas: {len(selected)} reglas")
    print(f"Salida: {args.output}")

    for r in selected:
        print(f"- {r['corruption_id']} | {r['type']} | {r['find']} -> {r['replace']}")


if __name__ == "__main__":
    main()