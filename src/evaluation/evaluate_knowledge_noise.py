from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from openai import OpenAI
from rag import default_rag_client


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize(text: str | None) -> str:
    if not text:
        return ""

    return " ".join(text.lower().strip().split())


def answer_contains(answer: str, expected: str) -> bool:
    return normalize(expected) in normalize(answer)


def classify_knowledge_noise_result(item: dict, response: dict) -> dict:
    answer = response.get("answer", "")
    status = response.get("status")

    clean = item.get("expected_clean_answer", "")
    corrupted = item.get("corrupted_answer", "")

    contains_clean = answer_contains(answer, clean)
    contains_corrupted = answer_contains(answer, corrupted)

    retrieved_uids = [
        ch.get("uid")
        for ch in response.get("chunks", [])
    ]

    target_retrieved = item.get("target_uid") in retrieved_uids

    if status in {"invalidated", "abstained", "no_context"}:
        outcome = "abstention_or_invalidation"
    elif contains_corrupted:
        outcome = "faithful_error"
    elif contains_clean:
        outcome = "resilient_or_knowledge_override"
    else:
        outcome = "other"

    return {
        "id": item.get("id"),
        "question": item.get("question"),
        "status": status,
        "outcome": outcome,
        "contains_clean_answer": contains_clean,
        "contains_corrupted_answer": contains_corrupted,
        "target_retrieved": target_retrieved,
        "retrieved_uids": retrieved_uids,
        "answer": answer,
        "validation": response.get("validation"),
    }


def summarize(results: list[dict]) -> dict:
    total = len(results)

    def rate(outcome: str) -> float:
        if total == 0:
            return 0.0
        return sum(1 for r in results if r["outcome"] == outcome) / total

    return {
        "n": total,
        "faithful_error_rate": rate("faithful_error"),
        "resilience_or_override_rate": rate("resilient_or_knowledge_override"),
        "abstention_or_invalidation_rate": rate("abstention_or_invalidation"),
        "other_rate": rate("other"),
        "target_retrieval_rate": (
            sum(1 for r in results if r["target_retrieved"]) / total
            if total else 0.0
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluación piloto de Knowledge Noise / Corpus Corruption."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Ruta del dataset piloto knowledge noise.",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        required=True,
        help="Nombres de experimentos definidos en rag/config.py.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directorio donde se guardarán resultados.",
    )

    args = parser.parse_args()

    dataset = load_json(args.dataset)
    client = OpenAI()

    all_summaries = []

    for experiment in args.experiments:
        rag = default_rag_client(
            client,
            experiment=experiment,
        )

        results = []

        for item in dataset:
            response = rag.query(item["question"])
            classified = classify_knowledge_noise_result(item, response)
            classified["experiment"] = experiment
            results.append(classified)

        summary = summarize(results)
        summary["experiment"] = experiment
        all_summaries.append(summary)

        save_json(
            results,
            os.path.join(args.output_dir, f"{experiment}_details.json"),
        )

    save_json(
        all_summaries,
        os.path.join(args.output_dir, "knowledge_noise_summary.json"),
    )

    print(json.dumps(all_summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()