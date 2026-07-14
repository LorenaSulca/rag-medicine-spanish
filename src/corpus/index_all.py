from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", default="../data/chunks")
    parser.add_argument("--embedding-models", nargs="+", default=["openai", "e5", "medcpt"])
    parser.add_argument("--include-flat", action="store_true")
    parser.add_argument("--reset-first", action="store_true")
    args = parser.parse_args()

    chunks_dir = Path(args.chunks_dir)
    section_files = sorted(chunks_dir.glob("*_chunks_sectioned_entities.json"))
    flat_files = sorted(chunks_dir.glob("*_chunks_flat_entities.json")) if args.include_flat else []

    jobs = []

    for path in section_files:
        doc_id = path.name.replace("_chunks_sectioned_entities.json", "")
        for model in args.embedding_models:
            jobs.append((path.name, doc_id, "sections", model))

    for path in flat_files:
        doc_id = path.name.replace("_chunks_flat_entities.json", "")
        for model in args.embedding_models:
            jobs.append((path.name, doc_id, "flat", model))

    seen_variants = set()

    for chunks_file, doc_id, chunking_variant, model in jobs:
        variant = f"{chunking_variant}_{model}"
        reset = args.reset_first and variant not in seen_variants
        seen_variants.add(variant)

        cmd = [
            sys.executable,
            "-m",
            "corpus.index_faiss",
            chunks_file,
            doc_id,
            "--chunking-variant",
            chunking_variant,
            "--embedding-model",
            model,
        ]

        if reset:
            cmd.append("--reset")

        run(cmd)

    print("\nIndexación completa.")


if __name__ == "__main__":
    main()