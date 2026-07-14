from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print("\n> " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_dirs(*dirs: Path) -> None:
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def document_id_from_txt(path: Path) -> str:
    return path.stem.lower().strip()


def main():
    parser = argparse.ArgumentParser(
        description="Genera secciones, chunks sectioned/flat y alinea entidades para todos los TXT."
    )

    parser.add_argument(
        "--text-dir",
        default="../data/text",
        help="Carpeta con TXT limpios.",
    )

    parser.add_argument(
        "--entities-dir",
        default="../data/medspaner",
        help="Carpeta con entidades MEDSPANER.",
    )

    parser.add_argument(
        "--sections-dir",
        default="../data/sections",
        help="Carpeta de salida de secciones.",
    )

    parser.add_argument(
        "--chunks-tmp-dir",
        default="../data/chunks_tmp",
        help="Carpeta temporal de chunks sin entidades.",
    )

    parser.add_argument(
        "--chunks-dir",
        default="../data/chunks",
        help="Carpeta final de chunks con entidades.",
    )

    parser.add_argument(
        "--chunk-size",
        default=None,
        type=int,
        help="Tamaño de chunk opcional.",
    )

    parser.add_argument(
        "--overlap",
        default=None,
        type=int,
        help="Overlap opcional.",
    )

    parser.add_argument(
        "--skip-flat",
        action="store_true",
        help="No generar chunks flat.",
    )

    parser.add_argument(
        "--skip-sections",
        action="store_true",
        help="No generar chunks por secciones.",
    )

    args = parser.parse_args()

    text_dir = Path(args.text_dir)
    entities_dir = Path(args.entities_dir)
    sections_dir = Path(args.sections_dir)
    chunks_tmp_dir = Path(args.chunks_tmp_dir)
    chunks_dir = Path(args.chunks_dir)

    ensure_dirs(sections_dir, chunks_tmp_dir, chunks_dir)

    txt_files = sorted(text_dir.glob("*.txt"))

    if not txt_files:
        raise FileNotFoundError(f"No se encontraron TXT en: {text_dir}")

    for txt_path in txt_files:
        document_id = document_id_from_txt(txt_path)
        entities_path = entities_dir / f"{document_id}_entities.json"

        if not entities_path.exists():
            print(f"\n[SKIP] No existe entities para {document_id}: {entities_path}")
            continue

        print("\n" + "=" * 80)
        print(f"Procesando: {document_id}")
        print("=" * 80)

        sections_path = sections_dir / f"{document_id}_sections.json"

        section_chunks_tmp = chunks_tmp_dir / f"{document_id}_chunks_sectioned.json"
        section_chunks_final = chunks_dir / f"{document_id}_chunks_sectioned_entities.json"

        flat_chunks_tmp = chunks_tmp_dir / f"{document_id}_chunks_flat.json"
        flat_chunks_final = chunks_dir / f"{document_id}_chunks_flat_entities.json"

        # 1. Extraer secciones
        run([
            sys.executable,
            "-m",
            "corpus.section_extractor",
            "--input",
            str(txt_path),
            "--document-id",
            document_id,
            "--output",
            str(sections_path),
        ])

        # 2. Section chunks + entity aligner
        if not args.skip_sections:
            cmd = [
                sys.executable,
                "-m",
                "corpus.section_chunker",
                "--input",
                str(sections_path),
                "--output",
                str(section_chunks_tmp),
            ]

            if args.chunk_size is not None:
                cmd.extend(["--chunk-size", str(args.chunk_size)])

            if args.overlap is not None:
                cmd.extend(["--overlap", str(args.overlap)])

            run(cmd)

            run([
                sys.executable,
                "-m",
                "corpus.entity_aligner",
                "--chunks",
                str(section_chunks_tmp),
                "--entities",
                str(entities_path),
                "--output",
                str(section_chunks_final),
            ])

        # 3. Flat chunks + entity aligner
        if not args.skip_flat:
            cmd = [
                sys.executable,
                "-m",
                "corpus.flat_chunker",
                "--input",
                str(txt_path),
                "--document-id",
                document_id,
                "--sections",
                str(sections_path),
                "--output",
                str(flat_chunks_tmp),
            ]

            if args.chunk_size is not None:
                cmd.extend(["--chunk-size", str(args.chunk_size)])

            if args.overlap is not None:
                cmd.extend(["--overlap", str(args.overlap)])

            run(cmd)

            run([
                sys.executable,
                "-m",
                "corpus.entity_aligner",
                "--chunks",
                str(flat_chunks_tmp),
                "--entities",
                str(entities_path),
                "--output",
                str(flat_chunks_final),
            ])

    print("\nProceso completo.")


if __name__ == "__main__":
    main()