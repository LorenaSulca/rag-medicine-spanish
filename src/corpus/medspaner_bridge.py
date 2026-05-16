import os
import json
import subprocess
import tempfile

from retrieval.utils_env import (
    get_old_python,
    get_medspaner_script,
    get_medspaner_config,
)


def project_root() -> str:
    return os.path.dirname(os.getcwd())


def run_medspaner_question(texto: str):
    old_python = project_root() + get_old_python()
    medspaner_script = project_root() + get_medspaner_script()
    medspaner_config = project_root() + get_medspaner_config()

    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        suffix=".txt",
        encoding="utf-8"
    ) as tmp:
        tmp.write(texto)
        tmp_path = tmp.name

    json_output = os.path.abspath(os.path.join(
        os.path.dirname(medspaner_script),
        "medspaner_output.json"
    ))

    if os.path.exists(json_output):
        os.remove(json_output)

    cmd = [
        old_python,
        medspaner_script,
        "-conf", medspaner_config,
        "-input", tmp_path,
    ]

    result = subprocess.run(
        cmd,
        cwd=os.path.abspath(os.path.dirname(medspaner_script)),
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if result.returncode != 0:
        print("Error ejecutando MEDSPANER:")
        print(result.stderr)
        return []

    if os.path.exists(json_output):
        try:
            with open(json_output, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print("Error leyendo JSON de MEDSPANER:", e)
            return []

    return []


def run_medspaner_prospect(input_path: str, output_path: str) -> bool:
    old_python = project_root() + get_old_python()
    medspaner_script = project_root() + get_medspaner_script()
    medspaner_config = project_root() + get_medspaner_config()

    medspaner_root = os.path.abspath(os.path.dirname(medspaner_script))
    internal_json = os.path.join(medspaner_root, "medspaner_output.json")

    if os.path.exists(internal_json):
        os.remove(internal_json)

    cmd = [
        old_python,
        medspaner_script,
        "-conf", medspaner_config,
        "-input", input_path,
    ]

    print("\nEjecutando MEDSPANER")
    print("CMD:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=medspaner_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    if result.returncode != 0:
        print("Error ejecutando MEDSPANER:")
        print(result.stderr)
        return False

    if not os.path.exists(internal_json):
        print("MEDSPANER no generó medspaner_output.json")
        return False

    try:
        with open(internal_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Error leyendo JSON generado:", e)
        return False

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_file": input_path,
                "output_file": output_path,
                "entities": data,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("MEDSPANER finalizó correctamente.")
    print(f"JSON guardado en: {output_path}")
    return True