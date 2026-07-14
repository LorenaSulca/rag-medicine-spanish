from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-per-document", type=int, default=65)
    args = parser.parse_args()

    merged = []

    for input_path in args.inputs:
        data = load_json(input_path)

        if not isinstance(data, list):
            raise ValueError(f"No es lista JSON: {input_path}")

        doc_id = data[0]["document_id"] if data else Path(input_path).stem
        selected = data[: args.limit_per_document]

        for idx, item in enumerate(selected, start=1):
            item = dict(item)
            item["id"] = f"{doc_id}_{idx:03d}"
            merged.append(item)

    save_json(args.output, merged)

    print(f"Dataset final: {args.output}")
    print(f"Total QA: {len(merged)}")


if __name__ == "__main__":
    main()