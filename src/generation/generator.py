from openai import OpenAI


def build_base_prompt(question: str, chunks: list):
    ctx_text = "\n\n".join([c["text"] for c in chunks])

    prompt = f"""
Usa EXCLUSIVAMENTE el contexto para responder.

Si la información no está en el contexto, responde:
"No se puede responder con la información disponible en el contexto proporcionado."

Pregunta:
{question}

Contexto:
{ctx_text}

Respuesta:
"""
    return prompt


def generate_answer(
    llm_client: OpenAI,
    question: str,
    chunks: list,
    signals=None,
    citation_prompt: bool = False,
):
    """
    Generador base (sin citas aún).
    """

    prompt = build_base_prompt(question, chunks)

    resp = llm_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}],
    )

    return resp.choices[0].message.content.strip()