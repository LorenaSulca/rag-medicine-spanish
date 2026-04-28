from openai import OpenAI


ABSTENTION_MESSAGE = (
    "No se puede responder con la información disponible en el contexto proporcionado."
)


def build_context_block(chunks: list) -> str:
    """
    Construye el contexto numerado para que el modelo pueda citar fuentes.
    """
    lines = []

    for i, ch in enumerate(chunks, start=1):
        document_id = ch.get("document_id", "desconocido")
        chunk_id = ch.get("chunk_id", "?")
        text = ch.get("text", "").strip()

        source_id = f"Fuente {i} | documento: {document_id} | chunk: {chunk_id}"

        lines.append(f"[{source_id}]")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def build_base_prompt(question: str, chunks: list) -> str:
    ctx_text = "\n\n".join([c.get("text", "") for c in chunks])

    return f"""
Usa EXCLUSIVAMENTE el contexto para responder.

Si la información no está en el contexto, responde exactamente:
"{ABSTENTION_MESSAGE}"

Pregunta:
{question}

Contexto:
{ctx_text}

Respuesta:
"""


def build_citation_prompt(question: str, chunks: list, signals=None) -> str:
    context_block = build_context_block(chunks)

    meds = ", ".join((signals or {}).get("meds", [])) or "no detectado"
    diso = ", ".join((signals or {}).get("diso", [])) or "no detectado"
    forms = ", ".join((signals or {}).get("forms", [])) or "no detectado"

    return f"""
Eres un asistente especializado en información farmacéutica.

Tu tarea es responder usando EXCLUSIVAMENTE el contexto proporcionado.

REGLAS ESTRICTAS:
- No uses conocimiento externo.
- No inventes dosis, indicaciones, contraindicaciones, efectos adversos ni recomendaciones.
- Cada afirmación médica relevante debe incluir una cita al fragmento que la respalda.
- Usa el formato de cita: [Fuente N].
- Si una afirmación no puede citarse con una fuente del contexto, no la incluyas.
- Si ninguna parte de la pregunta puede responderse con el contexto, responde exactamente:
"{ABSTENTION_MESSAGE}"
- Si solo una parte puede responderse, responde únicamente esa parte y menciona que no se encontró información suficiente para lo demás.
- No cites fuentes que no respalden directamente la afirmación.

Pregunta del usuario:
{question}

Entidades detectadas en la pregunta:
- Medicamentos: {meds}
- Patologías / síntomas: {diso}
- Formas / vías farmacéuticas: {forms}

=== CONTEXTO INICIO ===
{context_block}
=== CONTEXTO FIN ===

Responde de forma clara, breve y estructurada.

FORMATO OBLIGATORIO:
- Responde en viñetas.
- Cada viñeta debe contener una sola afirmación médica principal.
- Cada viñeta debe terminar con una o más citas en formato [Fuente N].
- No agrupes todas las citas al final de la respuesta.
- No incluyas afirmaciones sin cita.
"""


def generate_answer(
    llm_client: OpenAI,
    question: str,
    chunks: list,
    signals=None,
    citation_prompt: bool = False,
):
    """
    Genera respuesta usando prompt base o prompt con citas obligatorias.
    """

    if citation_prompt:
        prompt = build_citation_prompt(
            question=question,
            chunks=chunks,
            signals=signals,
        )
    else:
        prompt = build_base_prompt(
            question=question,
            chunks=chunks,
        )

    resp = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    return resp.choices[0].message.content.strip()