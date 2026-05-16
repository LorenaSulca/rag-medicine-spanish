import json
import re
from openai import OpenAI


CATEGORIES = [
    "indications",
    "dosage",
    "contraindications",
    "interactions",
    "pregnancy_lactation",
    "adverse_effects",
    "overdose",
    "administration",
    "storage",
    "warnings",
    "out_of_context",
]


def safe_json_loads(text: str):
    """
    Extrae un JSON aunque el modelo devuelva texto alrededor.
    """
    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No se encontró una lista JSON válida en la respuesta.")
    return json.loads(match.group(0))


def build_qa_prompt(
    document_id: str,
    section_name: str,
    section_text: str,
    n: int = 5,
) -> str:
    return f"""
Eres un experto en farmacología y evaluación de sistemas RAG médicos.

Genera EXACTAMENTE {n} pares pregunta-respuesta basados SOLO en el texto entregado.

Cada elemento debe servir para evaluar un sistema RAG médico.

REGLAS:
- Las preguntas deben ser contestables únicamente con el texto proporcionado.
- La respuesta ground_truth debe ser factual, breve y verificable.
- No inventes información.
- No uses conocimiento externo.
- Incluye variedad de dificultad: easy, medium y hard cuando sea posible.
- Si la pregunta requiere dosis, cantidades, frecuencias o tiempos, marca requires_numeric_grounding=true.
- Si la pregunta necesita combinar más de una parte del texto, marca requires_multi_hop=true.
- expected_behavior debe ser "answered", salvo que explícitamente sea una pregunta parcialmente respondible.

Categorías permitidas:
{CATEGORIES}

Devuelve EXCLUSIVAMENTE un JSON válido con esta forma:

[
  {{
    "document_id": "{document_id}",
    "category": "dosage",
    "difficulty": "medium",
    "question": "...",
    "ground_truth": "...",
    "source_sections": ["{section_name}"],
    "expected_behavior": "answered",
    "requires_numeric_grounding": true,
    "requires_multi_hop": false,
    "notes": null
  }}
]

DOCUMENTO:
{document_id}

SECCIÓN:
{section_name}

TEXTO:
\"\"\"
{section_text}
\"\"\"
"""


def build_out_of_context_prompt(
    document_id: str,
    known_sections: list[str],
    n: int = 3,
) -> str:
    return f"""
Eres un experto en evaluación de sistemas RAG médicos.

Genera EXACTAMENTE {n} preguntas fuera de contexto para el documento "{document_id}".

Estas preguntas deben parecer plausibles para un usuario, pero NO deben poder responderse
con un prospecto farmacológico típico de este medicamento.

Objetivo:
- Evaluar si el sistema sabe abstenerse correctamente.

REGLAS:
- expected_behavior debe ser "abstained".
- ground_truth debe ser exactamente:
"No se puede responder con la información disponible en el prospecto."
- No hagas preguntas peligrosas ni instrucciones clínicas personalizadas.
- No preguntes por información que normalmente sí aparece en prospectos, como dosis, efectos adversos o contraindicaciones.

Secciones disponibles del documento:
{known_sections}

Devuelve EXCLUSIVAMENTE un JSON válido:

[
  {{
    "document_id": "{document_id}",
    "category": "out_of_context",
    "difficulty": "hard",
    "question": "...",
    "ground_truth": "No se puede responder con la información disponible en el prospecto.",
    "source_sections": [],
    "expected_behavior": "abstained",
    "requires_numeric_grounding": false,
    "requires_multi_hop": false,
    "notes": "Pregunta fuera de contexto para evaluar abstención."
  }}
]
"""


def generate_qa_for_section(
    client: OpenAI,
    document_id: str,
    section_name: str,
    section_text: str,
    n: int = 5,
    model: str = "gpt-4o-mini",
):
    prompt = build_qa_prompt(
        document_id=document_id,
        section_name=section_name,
        section_text=section_text,
        n=n,
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    return safe_json_loads(resp.choices[0].message.content)


def generate_out_of_context_qa(
    client: OpenAI,
    document_id: str,
    known_sections: list[str],
    n: int = 3,
    model: str = "gpt-4o-mini",
):
    prompt = build_out_of_context_prompt(
        document_id=document_id,
        known_sections=known_sections,
        n=n,
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
    )

    return safe_json_loads(resp.choices[0].message.content)


def assign_ids(items: list[dict], prefix: str = "qa") -> list[dict]:
    for idx, item in enumerate(items, start=1):
        item["id"] = f"{prefix}_{idx:04d}"
    return items