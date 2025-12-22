import os
import json
import argparse
import textwrap
from openai import OpenAI
from utils_env import get_openai_api_key, get_data_dir, get_QA_dir

client = OpenAI(api_key=get_openai_api_key())

def clean_section_name(name):
    name = name.lower().strip()
    accents = {
        "á": "a", "é": "e", "í": "i",
        "ó": "o", "ú": "u", "ñ": "n"
    }
    for a, b in accents.items():
        name = name.replace(a, b)
    return name


def robust_json_extract(raw_text, label="QA"):
    """
    Extrae JSON válido desde la respuesta del modelo.
    """
    try:
        start = raw_text.index("[")
        end = raw_text.rindex("]") + 1
        json_str = raw_text[start:end]
        return json.loads(json_str)
    except Exception:
        print(f"ERROR: No se pudo extraer JSON válido ({label}). RAW:\n{raw_text}\n")
        return []


def parse_score_json(raw):
    """
    Espera un JSON del tipo:
    {
        "faithful": true/false,
        "score": 0.0–1.0
    }
    """
    try:
        data = json.loads(raw)
        if "score" in data:
            return float(data["score"])
        return 0.0
    except Exception:
        return 0.0

def generate_QA_from_section(section_name, section_text, n=10):
    prompt = f"""
Eres un asistente experto en farmacología.

Genera EXACTAMENTE {n} pares de pregunta-respuesta (QA) basados SOLO en este texto de prospecto.
Cada QA debe ser RELEVANTE, FACTUAL, y contestable únicamente con este texto.

Devuelve EXCLUSIVAMENTE un JSON con el formato:

[
  {{"question": "...", "answer": "..."}},
  ...
]

SECCIÓN: "{section_name}"

TEXTO:
\"\"\"
{section_text}
\"\"\"
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=2000
    )

    raw = response.choices[0].message.content.strip()
    return robust_json_extract(raw, label=section_name)


def validate_QA_pair(question, answer):
    """
    Valida que la respuesta sea fiel al texto base.
    """
    prompt = f"""
Evalúa si la respuesta es fiel al contenido del prospecto.

Devuelve SOLO este JSON, nada más:
{{
  "faithful": <true|false>,
  "score": <float entre 0 y 1>
}}

Pregunta: "{question}"
Respuesta: "{answer}"
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200
    )

    raw = resp.choices[0].message.content.strip()

    return parse_score_json(raw)

def split_sections(text):
    """
    Extrae secciones relevantes por títulos.
    """
    titles = [
        "conservación", "conservacion",
        "advertencias",
        "composición", "composicion"
    ]

    sections = {}
    lines = text.split("\n")

    current_title = None
    buffer = []

    for line in lines:
        clean = clean_section_name(line)

        if any(clean.startswith(t) for t in titles):
            if current_title and buffer:
                sections[current_title] = "\n".join(buffer)
                buffer = []

            current_title = clean.split()[0]
        else:
            if current_title:
                buffer.append(line)

    if current_title and buffer:
        sections[current_title] = "\n".join(buffer)

    return sections


def generate_dataset(input_txt, output_json, test_mode=False):
    DATA_DIR = get_data_dir()
    QA_DIR = get_QA_dir()
    input_path = os.path.join(DATA_DIR, input_txt)
    output_path_json = os.path.join(QA_DIR, output_json)
    output_path_csv = output_path_json.replace(".json", ".csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    text = open(input_path, "r", encoding="utf-8").read()

    sections = split_sections(text)
    print(f"Se encontraron {len(sections)} secciones relevantes.\n")

    all_QA = []

    for sec_name, sec_text in sections.items():
        print(f"GENERANDO QA PARA SECCIÓN: {sec_name.upper()}")

        Q = generate_QA_from_section(sec_name, sec_text, n=3 if test_mode else 10)
        print(f"{len(Q)} QA generados. Validando")

        for qa in Q:
            q = qa["question"]
            a = qa["answer"]

            score = validate_QA_pair(q, a)

            if score >= 0.6:
                print(f"ACEPTADO ({score:.2f}) - {q[:50]}")
                all_QA.append({"question": q, "answer": a})
            else:
                print(f"DESCARTADO ({score:.2f}) - {q[:50]}")

    # Guardar JSON final
    os.makedirs(os.path.dirname(output_path_json), exist_ok=True)
    with open(output_path_json, "w", encoding="utf-8") as f:
        json.dump(all_QA, f, indent=2, ensure_ascii=False)

    # Guardar CSV
    with open(output_path_csv, "w", encoding="utf-8") as f:
        f.write("question,answer\n")
        for qa in all_QA:
            f.write(f"\"{qa['question']}\",\"{qa['answer']}\"\n")

    print(f"DATASET GENERADO CON {len(all_QA)} QA VALIDOS")
    print(f"JSON → {output_path_json}")
    print(f"CSV  → {output_path_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generador de QA dataset desde prospectos farmacológicos")
    parser.add_argument("input_txt", help="Archivo TXT limpio desde /data/")
    parser.add_argument("output_json", help="Archivo de salida (JSON)")
    parser.add_argument("--test", action="store_true", help="Modo rápido: 3 QA por sección")

    args = parser.parse_args()

    generate_dataset(
        input_txt=args.input_txt,
        output_json=args.output_json,
        test_mode=args.test
    )
