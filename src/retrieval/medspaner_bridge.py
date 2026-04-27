import os
import json
import subprocess
import argparse
import tempfile
from .utils_env import (
    get_old_python,
    get_medspaner_script,
    get_medspaner_config,
    get_data_dir
)

def run_medspaner_question(texto: str):

    old_python = os.path.dirname(os.getcwd()) +get_old_python()
    medspaner_script = os.path.dirname(os.getcwd()) + get_medspaner_script()
    medspaner_config = os.path.dirname(os.getcwd()) + get_medspaner_config()

    # Archivo temporal con la pregunta
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".txt", encoding="utf-8"
    ) as tmp:
        tmp.write(texto)
        tmp_path = tmp.name

    # Path al JSON generado por MEDSPANER
    json_output = os.path.abspath(os.path.join(
        os.path.dirname(medspaner_script),
        "medspaner_output.json"
    ))

    # Borrar JSON previo si existe
    if os.path.exists(json_output):
        os.remove(json_output)

    # Construir comando CLI
    cmd = [
        old_python,
        medspaner_script,
        "-conf", medspaner_config,
        "-input", tmp_path
    ]

    # Ejecutar MEDSPANER
    result = subprocess.run(
        cmd,
        cwd=os.path.abspath(os.path.join(os.path.dirname(medspaner_script))),
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    # Borrar archivo temporal
    try:
        os.remove(tmp_path)
    except:
        pass

    # Verificar error de ejecución
    if result.returncode != 0:
        print("Error ejecutando MEDSPANER:")
        print(result.stderr)
        return []

    # Cargar JSON generado por MEDSPANER
    if os.path.exists(json_output):
        try:
            with open(json_output, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print("Error leyendo JSON de MEDSPANER:", e)
            return []

    # Fallback (por si JSON no fue creado)
    return []


def run_medspaner_prospect(input_path: str, output_path: str):

    old_python = os.path.dirname(os.getcwd()) + get_old_python() 
    medspaner_script = os.path.dirname(os.getcwd()) + get_medspaner_script()
    medspaner_config = os.path.dirname(os.getcwd()) + get_medspaner_config()

    # Directorio root de MEDSPANER (para cwd)
    medspaner_root = os.path.abspath(os.path.join(os.path.dirname(medspaner_script)))

    # JSON que genera MEDSPANER internamente
    internal_json = os.path.join(medspaner_root, "medspaner_output.json")

    # Borrar el JSON si existe
    if os.path.exists(internal_json):
        os.remove(internal_json)

    # Construir comando
    cmd = [
        old_python,
        medspaner_script,
        "-conf", medspaner_config,
        "-input", input_path
    ]

    print("\n Ejecutando MEDSPANER")
    print("CMD:", " ".join(cmd))

    result = subprocess.run(
        cmd,
        cwd=medspaner_root,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    if result.returncode != 0:
        print("Error ejecutando MEDSPANER:")
        print(result.stderr)
        return False

    # Cargar el JSON interno
    if not os.path.exists(internal_json):
        print("MEDSPANER no genero resultado_medspaner.json")
        return False

    try:
        with open(internal_json, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Error leyendo JSON generado:", e)
        return False
    # Guardar en el JSON final si se evalua prospecto
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "input_file": input_path,
            "output_file": output_path,
            "entities": data
        }, f, ensure_ascii=False, indent=2)

    print("MEDSPANER finalizó correctamente.")
    print(f"JSON guardado en: {output_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bridge para ejecutar MEDSPANER desde Python no 3.7")
    parser.add_argument("input_txt", help="Nombre del archivo de entrada (dentro de /data/)")
    parser.add_argument("output_json", help="Nombre del json final (dentro de /data/)")

    args = parser.parse_args()

    data_dir = get_data_dir()
    input_path = os.path.dirname(os.getcwd()) + data_dir + args.input_txt
    output_path = os.path.dirname(os.getcwd()) + data_dir +  args.output_json

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"El archivo {input_path} no existe")

    run_medspaner_prospect(input_path, output_path)
