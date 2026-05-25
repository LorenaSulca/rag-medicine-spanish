import argparse
import csv
import json
import os
from statistics import mean
from typing import Any, Optional

from openai import OpenAI
from rag import default_rag_client


DEFAULT_EXPERIMENTS = [
    "baseline_flat",
    "baseline_sections",
    "propuesta_1_full_flat",
    "propuesta_1_full_sections",
    "propuesta_2_full_flat",
    "propuesta_2_full_sections",
]

SAFE_REJECTION_STATUSES = {"abstained", "invalidated", "no_context"}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path: str, rows: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not rows:
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_section_name(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.lower().strip().split())


def get_chunk_sections(chunks: list[dict]) -> list[str]:
    return [
        normalize_section_name(ch.get("section_name"))
        for ch in chunks
        if ch.get("section_name")
    ]


def compute_section_retrieval_metrics(item: dict, chunks: list[dict]) -> dict:
    """
    Métricas determinísticas basadas en secciones esperadas.

    context_precision_section:
      proporción de chunks recuperados cuya sección pertenece a las secciones esperadas.

    context_recall_section:
      proporción de secciones esperadas que aparecen al menos una vez entre los chunks.

    context_relevance_section:
      1 si al menos un chunk proviene de una sección esperada, 0 si no.
    """

    expected_sections = [
        normalize_section_name(s)
        for s in item.get("source_sections", [])
        if s
    ]

    chunk_sections = get_chunk_sections(chunks)

    if not expected_sections:
        return {
            "context_precision_section": None,
            "context_recall_section": None,
            "context_relevance_section": None,
        }

    if not chunk_sections:
        return {
            "context_precision_section": 0.0,
            "context_recall_section": 0.0,
            "context_relevance_section": 0.0,
        }

    expected_set = set(expected_sections)
    chunk_set = set(chunk_sections)

    relevant_chunks = sum(1 for s in chunk_sections if s in expected_set)
    retrieved_expected_sections = len(expected_set.intersection(chunk_set))

    precision = relevant_chunks / len(chunk_sections)
    recall = retrieved_expected_sections / len(expected_set)
    relevance = 1.0 if retrieved_expected_sections > 0 else 0.0

    return {
        "context_precision_section": precision,
        "context_recall_section": recall,
        "context_relevance_section": relevance,
    }


def compute_behavior_metrics(item: dict, status: str) -> dict:
    expected = item.get("expected_behavior")

    is_safe_rejection = status in SAFE_REJECTION_STATUSES

    exact_behavior_match = expected == status

    correct_answered = expected == "answered" and status == "answered"
    correct_partial = expected == "partial" and status == "partial"
    correct_abstention = expected == "abstained" and is_safe_rejection

    behavior_correct = correct_answered or correct_partial or correct_abstention

    wrong_abstention = expected == "answered" and is_safe_rejection
    missed_abstention = expected == "abstained" and status == "answered"

    return {
        "expected_behavior": expected,
        "behavior_correct": behavior_correct,
        "exact_behavior_match": exact_behavior_match,
        "correct_abstention": correct_abstention,
        "wrong_abstention": wrong_abstention,
        "missed_abstention": missed_abstention,
    }


def extract_validation_metrics(validation: Optional[dict]) -> dict:
    if not validation:
        return {
            "usr": None,
            "citation_consistency": None,
            "unsupported_count": None,
            "numeric_support_rate": None,
            "unsupported_numeric_count": None,
        }

    multilevel = validation.get("multilevel") or {}

    return {
        "usr": validation.get("usr"),
        "citation_consistency": validation.get("citation_consistency"),
        "unsupported_count": len(validation.get("unsupported_sentences", [])),
        "numeric_support_rate": multilevel.get("numeric_support_rate"),
        "unsupported_numeric_count": len(multilevel.get("unsupported_numeric_mentions", [])),
    }


def run_one(client: OpenAI, item: dict, experiment: str) -> dict:
    rag = default_rag_client(client, experiment=experiment)

    response = rag.query(item["question"])

    chunks = response.get("chunks", [])
    index_variants = sorted({
        c.get("index_variant")
        for c in chunks
        if c.get("index_variant")
    })

    chunking_strategies = sorted({
        c.get("chunking_strategy")
        for c in chunks
        if c.get("chunking_strategy")
    })
    validation = response.get("validation")
    status = response.get("status")

    retrieval_metrics = compute_section_retrieval_metrics(item, chunks)
    behavior_metrics = compute_behavior_metrics(item, status)
    validation_metrics = extract_validation_metrics(validation)

    return {
        "id": item.get("id"),
        "document_id": item.get("document_id"),
        "category": item.get("category"),
        "difficulty": item.get("difficulty"),
        "requires_numeric_grounding": item.get("requires_numeric_grounding"),
        "requires_multi_hop": item.get("requires_multi_hop"),
        "experiment": experiment,
        "question": item.get("question"),
        "ground_truth": item.get("ground_truth"),
        "answer": response.get("answer"),
        "status": status,
        "num_chunks": len(chunks),
        "chunk_ids": json.dumps([c.get("chunk_id") for c in chunks], ensure_ascii=False),
        "chunk_sections": json.dumps(get_chunk_sections(chunks), ensure_ascii=False),
        "index_variants": json.dumps(index_variants, ensure_ascii=False),
       "chunking_strategies": json.dumps(chunking_strategies, ensure_ascii=False),
        **retrieval_metrics,
        **behavior_metrics,
        **validation_metrics,
    }


def safe_mean(values: list):
    clean = [v for v in values if isinstance(v, (int, float)) and v is not None]
    return mean(clean) if clean else None


def summarize(rows: list[dict]) -> list[dict]:
    experiments = sorted(set(r["experiment"] for r in rows))
    summary = []

    for exp in experiments:
        exp_rows = [r for r in rows if r["experiment"] == exp]
        n = len(exp_rows)

        summary.append({
            "experiment": exp,
            "n": n,

            "answered_rate": sum(r["status"] == "answered" for r in exp_rows) / n,
            "partial_rate": sum(r["status"] == "partial" for r in exp_rows) / n,
            "abstained_rate": sum(r["status"] == "abstained" for r in exp_rows) / n,
            "invalidated_rate": sum(r["status"] == "invalidated" for r in exp_rows) / n,
            "error_rate": sum(r["status"] == "error" for r in exp_rows) / n,

            "behavior_accuracy": sum(bool(r["behavior_correct"]) for r in exp_rows) / n,
            "correct_abstention_rate": sum(bool(r["correct_abstention"]) for r in exp_rows) / n,
            "wrong_abstention_rate": sum(bool(r["wrong_abstention"]) for r in exp_rows) / n,
            "missed_abstention_rate": sum(bool(r["missed_abstention"]) for r in exp_rows) / n,

            "avg_context_precision_section": safe_mean([r["context_precision_section"] for r in exp_rows]),
            "avg_context_recall_section": safe_mean([r["context_recall_section"] for r in exp_rows]),
            "avg_context_relevance_section": safe_mean([r["context_relevance_section"] for r in exp_rows]),

            "avg_usr": safe_mean([r["usr"] for r in exp_rows]),
            "avg_citation_consistency": safe_mean([r["citation_consistency"] for r in exp_rows]),
            "avg_unsupported_count": safe_mean([r["unsupported_count"] for r in exp_rows]),
            "avg_numeric_support_rate": safe_mean([r["numeric_support_rate"] for r in exp_rows]),
            "avg_unsupported_numeric_count": safe_mean([r["unsupported_numeric_count"] for r in exp_rows]),
            "avg_num_chunks": safe_mean([r["num_chunks"] for r in exp_rows]),
        })

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta evaluación batch sobre configuraciones RAG."
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Ruta al dataset QA clasificado.",
    )

    parser.add_argument(
        "--output-dir",
        default="../data/results",
        help="Directorio de salida.",
    )

    parser.add_argument(
        "--experiments",
        nargs="*",
        default=DEFAULT_EXPERIMENTS,
        help="Lista de experimentos a ejecutar.",
    )

    args = parser.parse_args()

    dataset = load_json(args.dataset)
    client = OpenAI()

    rows = []

    for experiment in args.experiments:
        print(f"\n===== EXPERIMENTO: {experiment} =====")

        for idx, item in enumerate(dataset, start=1):
            print(f"[{idx}/{len(dataset)}] {item['id']} - {item['question']}")

            try:
                row = run_one(client, item, experiment)
            except Exception as exc:
                row = {
                    "id": item.get("id"),
                    "document_id": item.get("document_id"),
                    "category": item.get("category"),
                    "difficulty": item.get("difficulty"),
                    "requires_numeric_grounding": item.get("requires_numeric_grounding"),
                    "requires_multi_hop": item.get("requires_multi_hop"),
                    "experiment": experiment,
                    "question": item.get("question"),
                    "ground_truth": item.get("ground_truth"),
                    "answer": "",
                    "status": "error",
                    "num_chunks": 0,
                    "chunk_ids": "[]",
                    "chunk_sections": "[]",
                    "context_precision_section": None,
                    "context_recall_section": None,
                    "context_relevance_section": None,
                    "expected_behavior": item.get("expected_behavior"),
                    "behavior_correct": False,
                    "exact_behavior_match": False,
                    "correct_abstention": False,
                    "wrong_abstention": False,
                    "missed_abstention": False,
                    "usr": None,
                    "citation_consistency": None,
                    "unsupported_count": None,
                    "numeric_support_rate": None,
                    "unsupported_numeric_count": None,
                    "error": str(exc),
                }

            rows.append(row)

    summary_rows = summarize(rows)

    detailed_json = os.path.join(args.output_dir, "experiment_results_detailed.json")
    detailed_csv = os.path.join(args.output_dir, "experiment_results_detailed.csv")
    summary_json = os.path.join(args.output_dir, "experiment_results_summary.json")
    summary_csv = os.path.join(args.output_dir, "experiment_results_summary.csv")

    save_json(detailed_json, rows)
    save_csv(detailed_csv, rows)
    save_json(summary_json, summary_rows)
    save_csv(summary_csv, summary_rows)

    print("\n===== ARCHIVOS GENERADOS =====")
    print(detailed_json)
    print(detailed_csv)
    print(summary_json)
    print(summary_csv)


if __name__ == "__main__":
    main()