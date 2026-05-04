from openai import OpenAI


ABSTENTION_MESSAGE = (
    "No se puede responder con la información disponible en el contexto proporcionado."
)


def build_refine_context_block(chunks: list) -> str:
    lines = []

    for i, ch in enumerate(chunks, start=1):
        document_id = ch.get("document_id", "desconocido")
        chunk_id = ch.get("chunk_id", "?")
        text = ch.get("text", "").strip()

        lines.append(f"[Fuente {i} | documento: {document_id} | chunk: {chunk_id}]")
        lines.append(text)
        lines.append("")

    return "\n".join(lines)


def build_refine_prompt(question: str, answer: str, chunks: list) -> str:
    context_block = build_refine_context_block(chunks)

    return f"""
Eres un verificador y refinador de respuestas para un sistema RAG médico.

Tu tarea es revisar una respuesta preliminar y producir una versión final más segura, breve y fiel al contexto.

REGLAS ESTRICTAS:
- Usa EXCLUSIVAMENTE el contexto proporcionado.
- No agregues información nueva que no esté en el contexto.
- Elimina cualquier afirmación que no pueda respaldarse con una fuente.
- Corrige citas incorrectas si una fuente diferente respalda mejor la afirmación.
- Mantén solo afirmaciones médicas verificables.
- Si ninguna afirmación puede verificarse, responde exactamente:
"{ABSTENTION_MESSAGE}"
- Responde en viñetas.
- Cada viñeta debe contener una sola afirmación médica principal.
- Cada viñeta debe terminar con una o más citas en formato [Fuente N].
- No incluyas explicaciones sobre el proceso de revisión.

Pregunta original:
{question}

Respuesta preliminar:
{answer}

=== CONTEXTO INICIO ===
{context_block}
=== CONTEXTO FIN ===

Respuesta final refinada:
"""


def refine_answer(
    llm_client: OpenAI,
    question: str,
    answer: str,
    chunks: list,
) -> str:
    prompt = build_refine_prompt(
        question=question,
        answer=answer,
        chunks=chunks,
    )

    resp = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    return resp.choices[0].message.content.strip()