from openai import OpenAI
from utils_env import get_openai_api_key
from retrieval_faiss import retrieve_chunks

import textwrap


client = OpenAI(api_key=get_openai_api_key())

ABSTENTION_MESSAGE = (
    "No se puede responder con la información disponible en el contexto proporcionado."
)

NO_CONTEXT_MESSAGE = (
    "No se encontraron fragmentos relevantes en la base de prospectos para responder la pregunta."
)

INVALIDATED_MESSAGE = (
    "La respuesta generada no pudo ser validada contra el contexto disponible. "
    "No se puede responder con suficiente fiabilidad."
)


def build_system_prompt():
    return textwrap.dedent(f"""
    Eres un asistente especializado en información farmacéutica.

    Tu tarea es responder preguntas utilizando EXCLUSIVAMENTE el contexto proporcionado.

    REGLAS ESTRICTAS:
    - No puedes usar conocimiento externo.
    - No puedes inferir ni completar información faltante.
    - Está PROHIBIDO inventar:
        - dosis
        - indicaciones
        - contraindicaciones
        - efectos adversos
    - Toda afirmación que hagas debe estar explícitamente respaldada por el contexto.

    MANEJO DE PREGUNTAS PARCIALES:
    - Si la pregunta tiene una o varias partes, responde únicamente aquellas partes que estén explícitamente respaldadas por el contexto.
    - Si una parte de la pregunta NO está respaldada por el contexto, debes indicarlo claramente usando una frase como:
      "No se encontró esa información específica en el contexto proporcionado."
    - NO rechaces toda la respuesta si al menos una parte de la pregunta sí puede responderse con evidencia.
    - SOLO debes responder EXACTAMENTE:
      "{ABSTENTION_MESSAGE}"
      si ninguna parte de la pregunta puede responderse con el contexto.

    VALIDACIÓN INTERNA OBLIGATORIA:
    Antes de responder, debes verificar:
    1. ¿La información que voy a dar está en el contexto?
    2. ¿Puedo asociarla a uno o más fragmentos del contexto?
    3. ¿Estoy respondiendo solo lo que sí está respaldado?

    Si no puedes validar ninguna parte de la respuesta, debes abstenerte completamente.

    IMPORTANTE:
    - Si decides abstenerte completamente, NO agregues explicación adicional.
    - Si decides abstenerte completamente, NO cites fragmentos.
    - En ese caso devuelve SOLO la frase exacta indicada.
    - Si puedes responder parcialmente, deja claro qué parte sí está respaldada y qué parte no.

    FORMATO:
    - Respuesta clara y estructurada
    - Basada únicamente en el contexto
    - En español
    """)

def normalize_text_to_tokens(text: str) -> set[str]:
    """
    Normalización simple para comparación léxica.
    """
    if not text:
        return set()

    cleaned = (
        text.lower()
        .replace(".", " ")
        .replace(",", " ")
        .replace(";", " ")
        .replace(":", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace("[", " ")
        .replace("]", " ")
        .replace('"', " ")
        .replace("'", " ")
        .replace("¿", " ")
        .replace("?", " ")
        .replace("¡", " ")
        .replace("!", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )

    return {tok for tok in cleaned.split() if tok.strip()}


def is_abstention_answer(answer: str) -> bool:
    """
    Detecta si el modelo devolvió explícitamente la respuesta de abstención.
    """
    if not answer:
        return False

    return answer.strip() == ABSTENTION_MESSAGE


def validate_answer_against_context(answer, chunks):
    """
    Validación heurística ligera.
    Verifica si la respuesta comparte suficiente base léxica con el contexto.
    No garantiza grounding real, pero sirve como filtro inicial barato.
    """
    if not answer or not chunks:
        return False

    context_text = " ".join([c.get("text", "") for c in chunks])

    answer_tokens = normalize_text_to_tokens(answer)
    context_tokens = normalize_text_to_tokens(context_text)

    if len(answer_tokens) == 0:
        return False

    overlap = answer_tokens.intersection(context_tokens)
    ratio = len(overlap) / len(answer_tokens)

    return ratio > 0.3


def build_context_block(chunks):
    """
    Construye el bloque de contexto a partir de los chunks recuperados.
    Cada chunk se numera y se indica de qué prospecto proviene.
    """
    context_lines = []

    for i, ch in enumerate(chunks, start=1):
        med_id = ch.get("document_id", "desconocido")
        chunk_id = ch.get("chunk_id", "?")
        score = round(ch.get("score", 0.0), 3)
        text = ch.get("text", "").strip()

        header = (
            f"[Fuente {i} | documento: {med_id} | chunk: {chunk_id} | "
            f"score (similitud vectorial): {score}]"
        )
        context_lines.append(header)
        context_lines.append(text)
        context_lines.append("")

    return "\n".join(context_lines)


def build_user_prompt(question, chunks, signals):
    """
    Construye el mensaje de usuario:
    - pregunta original
    - entidades detectadas por MEDSPANER
    - bloque de contexto
    """
    context_block = build_context_block(chunks)

    meds = ", ".join(signals.get("meds", [])) or "no detectado"
    diso = ", ".join(signals.get("diso", [])) or "no detectado"
    forms = ", ".join(signals.get("forms", [])) or "no detectado"

    meta_info = textwrap.dedent(f"""
    Pregunta del usuario:
    \"\"\"{question}\"\"\"

    Entidades detectadas en la pregunta (MEDSPANER):
    - Medicamentos: {meds}
    - Patologías / síntomas (DISO): {diso}
    - Formas / vías farmacéuticas: {forms}

    A continuación se presentan fragmentos de prospectos de medicamentos.
    Usa únicamente esta información para responder.
    """)

    full_prompt = (
        meta_info
        + "\n\n=== CONTEXTO INICIO ===\n"
        + context_block
        + "\n=== CONTEXTO FIN ===\n\n"
        + "Con base en el contexto anterior, responde de forma clara, estructurada y concisa."
    )

    return full_prompt


def build_result(answer, chunks, signals, medspaner_raw, status):
    """
    Estructura uniforme de salida.
    status:
    - answered
    - abstained
    - invalidated
    - no_context
    """
    return {
        "answer": answer,
        "chunks": chunks,
        "signals": signals,
        "medspaner_raw": medspaner_raw,
        "status": status,
    }


def answer_question(question: str):
    """
    Flujo:
    1) Retrieval (FAISS + MEDSPANER)
    2) Construcción de prompt
    3) Llamada a GPT-4o-mini
    4) Detección de abstención
    5) Validación heurística post-generación
    6) Salida final con estado
    """

    # 1. Recuperar contexto
    chunks, signals, medspaner_raw = retrieve_chunks(question)

    if not chunks:
        return build_result(
            answer=NO_CONTEXT_MESSAGE,
            chunks=[],
            signals=signals,
            medspaner_raw=medspaner_raw,
            status="no_context",
        )

    # 2. Construir mensajes
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(question, chunks, signals)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 3. Generar respuesta
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=600,
    )

    answer = response.choices[0].message.content.strip()

    # 4. Detectar abstención explícita del modelo
    if is_abstention_answer(answer):
        return build_result(
            answer=ABSTENTION_MESSAGE,
            chunks=[],
            signals=signals,
            medspaner_raw=medspaner_raw,
            status="abstained",
        )

    # 5. Validación heurística post-generación
    is_valid = validate_answer_against_context(answer, chunks)

    if not is_valid:
        return build_result(
            answer=INVALIDATED_MESSAGE,
            chunks=[],
            signals=signals,
            medspaner_raw=medspaner_raw,
            status="invalidated",
        )

    # 6. Respuesta normal
    return build_result(
        answer=answer,
        chunks=chunks,
        signals=signals,
        medspaner_raw=medspaner_raw,
        status="answered",
    )