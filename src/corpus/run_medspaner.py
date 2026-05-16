import argparse
import os

from corpus.medspaner_bridge import run_medspaner_prospect


def abs_path(path: str) -> str:
    return os.path.abspath(path)


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta MEDSPANER sobre un TXT limpio de prospecto."
    )

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    input_path = abs_path(args.input)
    output_path = abs_path(args.output)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe el archivo de entrada: {input_path}")

    ok = run_medspaner_prospect(
        input_path=input_path,
        output_path=output_path,
    )

    if not ok:
        raise RuntimeError("MEDSPANER falló.")


if __name__ == "__main__":
    main()