from __future__ import annotations

import argparse
import json
from pathlib import Path
from copy import deepcopy


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply_corruption_rule(chunk: dict, rule: dict) -> tuple[dict, bool]:
    corrupted = deepcopy(chunk)

    text = corrupted.get("text", "")
    find = rule["find"]
    replace = rule["replace"]

    if find not in text:
        return corrupted, False

    corrupted["text"] = text.replace(find, replace)

    corrupted.setdefault("knowledge_noise", [])
    corrupted["knowledge_noise"].append({
        "corruption_id": rule["corruption_id"],
        "type": rule["type"],
        "find": find,
        "replace": replace,
        "expected_clean_answer": rule.get("expected_clean_answer"),
        "corrupted_answer": rule.get("corrupted_answer"),
    })

    corrupted["_is_corrupted"] = True
    corrupted["_corruption_ids"] = [
        item["corruption_id"]
        for item in corrupted["knowledge_noise"]
    ]

    return corrupted, True


def corrupt_chunks(chunks: list[dict], rules: list[dict]) -> tuple[list[dict], list[dict]]:
    output = deepcopy(chunks)
    report = []

    for rule in rules:
        target_uid = rule.get("target_uid")
        applied = False

        for idx, chunk in enumerate(output):
            if chunk.get("uid") != target_uid:
                continue

            corrupted_chunk, ok = apply_corruption_rule(chunk, rule)

            if ok:
                output[idx] = corrupted_chunk
                applied = True

            report.append({
                "corruption_id": rule["corruption_id"],
                "target_uid": target_uid,
                "applied": applied,
                "type": rule.get("type"),
                "find": rule.get("find"),
                "replace": rule.get("replace"),
            })

            break

        if not applied and not any(r["corruption_id"] == rule["corruption_id"] for r in report):
            report.append({
                "corruption_id": rule["corruption_id"],
                "target_uid": target_uid,
                "applied": False,
                "type": rule.get("type"),
                "find": rule.get("find"),
                "replace": rule.get("replace"),
                "reason": "target_uid no encontrado",
            })

    return output, report


def main():
    parser = argparse.ArgumentParser(
        description="Genera una versión corrupta de chunks para Knowledge Noise."
    )

    parser.add_argument("--input", required=True, help="Ruta del JSON de chunks original.")
    parser.add_argument("--rules", required=True, help="Ruta del JSON de reglas de corrupción.")
    parser.add_argument("--output", required=True, help="Ruta de salida del JSON corrupto.")
    parser.add_argument("--report", required=True, help="Ruta de salida del reporte de corrupción.")

    args = parser.parse_args()

    chunks = load_json(args.input)
    rules = load_json(args.rules)

    corrupted_chunks, report = corrupt_chunks(chunks, rules)

    save_json(corrupted_chunks, args.output)
    save_json(report, args.report)

    print("Knowledge Noise aplicado.")
    print(f"Chunks originales: {args.input}")
    print(f"Chunks corruptos: {args.output}")
    print(f"Reporte: {args.report}")

    applied_count = sum(1 for item in report if item.get("applied"))
    print(f"Reglas aplicadas: {applied_count}/{len(rules)}")


if __name__ == "__main__":
    main()