from openai import OpenAI
from utils_env import get_openai_api_key
from retrieval_faiss import retrieve_chunks

import textwrap


# Inicializar cliente OpenAI

client = OpenAI(api_key=get_openai_api_key())


# Construcción del prompt para GPT-4o-mini

def build_system_prompt():
    """
    Prompt de rol del sistema: asistente médico-farmacéutico.
    """
    return textwrap.dedent("""
    Eres un asistente virtual que responde preguntas sobre medicamentos
    usando únicamente la información contenida en los prospectos oficiales
    proporcionados en el contexto.

    Reglas importantes:
    - Responde SIEMPRE en español.
    - Basa tus respuestas ÚNICAMENTE en el contexto proporcionado.
    - Si la información no está explícitamente en el contexto, di claramente
      que no se puede responder con los datos disponibles.
    - No inventes dosis, indicaciones, contraindicaciones ni efectos adversos.
    - Si la pregunta es ambigua, menciona las posibles interpretaciones.
    - Si se habla de dosis, recalca que la decisión final debe tomarla un
      profesional de la salud y que la información no sustituye criterio médico.
    - Cuando cites información, menciona el nombre del medicamento y,
      si es posible, la sección (por ejemplo: “Posología”, “Advertencias”).
    """)


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

        header = f"[Fuente {i} | documento: {med_id} | chunk: {chunk_id} | score (similitud vectorial): {score}]"
        context_lines.append(header)
        context_lines.append(text)
        context_lines.append("")

    return "\n".join(context_lines)


def build_user_prompt(question, chunks, signals):
    """
    Construye el mensaje de usuario que verá el modelo:
    - pregunta original
    - breve resumen de señales extraídas por MEDSPANER (medicamento, etc.)
    - bloque de contexto (texto de prospectos)
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

    full_prompt = meta_info + "\n\n" + "=== CONTEXTO INICIO ===\n" + context_block + "\n=== CONTEXTO FIN ===\n\n" + \
        "Con base en el contexto anterior, responde de forma clara, estructurada y concisa."

    return full_prompt


# Función principal: pregunta → respuesta

def answer_question(question: str):
    """
    Orquesta el flujo:
    1) Retrieval (FAISS + MEDSPANER)
    2) Construcción de prompt
    3) Llamada a GPT-4o-mini
    4) Devuelve respuesta + chunks usados + señales
    """

    # 1. Recuperar contexto
    chunks, signals, medspaner_raw = retrieve_chunks(question)

    # si no hay contexto
    if not chunks:
        return {
            "answer": "No se encontraron fragmentos relevantes en la base de prospectos para responder la pregunta.",
            "chunks": [],
            "signals": signals,
            "medspaner_raw": medspaner_raw
        }

    # 2. Construir mensajes para ChatCompletion
    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(question, chunks, signals)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 3. Llamar a GPT-4o-mini
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.2,
        max_tokens=600
    )

    answer = response.choices[0].message.content.strip()

    # 4. Devolver todo junto
    return {
        "answer": answer,
        "chunks": chunks,
        "signals": signals,
        "medspaner_raw": medspaner_raw
    }
