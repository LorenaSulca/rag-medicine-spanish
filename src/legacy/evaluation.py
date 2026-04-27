import json
import argparse
import os
import re
from bert_score import score as bert_score_fn
from openai import OpenAI

from utils_env import get_QA_dir
from retrieval_faiss import retrieve_chunks


# Configuracion Openai

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Parsing

def safe_extract_float(text, default=0.0):
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if not match:
        return default
    try:
        data = json.loads(match.group(0))
        score = float(data.get("score", default))
    except:
        return default
    return max(0.0, min(1.0, score))


def run_llm(prompt):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()



# Metricas LLM-as-judge

def metric_faithfulness(question, answer, gt_answer):
    prompt = f"""
Evalúa SI la respuesta generada es fiel al ground truth.

Devuelve SOLO un JSON válido:
{{"score": número entre 0 y 1}}

Pregunta:
{question}

Ground truth:
{gt_answer}

Respuesta generada:
{answer}
"""
    return safe_extract_float(run_llm(prompt))


def metric_answer_relevance(question, answer):
    prompt = f"""
Evalúa la relevancia de la respuesta con respecto a la pregunta.

Devuelve SOLO:
{{"score": número entre 0 y 1}}

Pregunta:
{question}

Respuesta:
{answer}
"""
    return safe_extract_float(run_llm(prompt))


def metric_context_relevance(question, chunks):
    ctx = "\n\n".join([c["text"] for c in chunks])
    prompt = f"""
Evalúa la relevancia del contexto respecto a la pregunta.

Devuelve SOLO:
{{"score": número entre 0 y 1}}

Pregunta:
{question}

Contexto:
{ctx}
"""
    return safe_extract_float(run_llm(prompt))


def metric_context_precision(question, gt_answer, chunks):
    prompt = f"""
Evalúa la precisión del contexto recuperado respecto al ground truth.

Devuelve SOLO:
{{"score": número entre 0 y 1}}

Pregunta:
{question}

Ground truth:
{gt_answer}

Chunks:
{json.dumps(chunks, ensure_ascii=False)}
"""
    return safe_extract_float(run_llm(prompt))


def metric_context_recall(question, gt_answer, chunks):
    prompt = f"""
Evalúa el recall del contexto respecto al ground truth.

Devuelve SOLO:
{{"score": número entre 0 y 1}}

Pregunta:
{question}

Ground truth:
{gt_answer}

Chunks:
{json.dumps(chunks, ensure_ascii=False)}
"""
    return safe_extract_float(run_llm(prompt))


# Evaluar una pregunta

def evaluate_one_question(question, gt_answer, verbose=False):
    # Retrieval
    chunks, signals, _ = retrieve_chunks(question)

    ctx_text = "\n\n".join([c["text"] for c in chunks])

    # Generar respuesta
    ans_prompt = f"""
Usa EXCLUSIVAMENTE el contexto para responder.

Pregunta:
{question}

Contexto:
{ctx_text}

Respuesta:
"""
    answer = run_llm(ans_prompt)

    if verbose:
        print("\n=== Pregunta ===", question)
        print("\n--- Respuesta generada ---")
        print(answer)
        print("\n--- Chunks recuperados ---")
        for c in chunks:
            print(c["chunk_id"], "score=", c["score"])

    # Métricas
    faith = metric_faithfulness(question, answer, gt_answer)
    ans_rel = metric_answer_relevance(question, answer)
    ctx_rel = metric_context_relevance(question, chunks)
    ctx_prec = metric_context_precision(question, gt_answer, chunks)
    ctx_rec = metric_context_recall(question, gt_answer, chunks)

    P, R, F1 = bert_score_fn(
        [answer], [gt_answer],
        lang="es",
        rescale_with_baseline=True
    )

    return answer, {
        "faithfulness": faith,
        "answer_relevance": ans_rel,
        "context_relevance": ctx_rel,
        "context_precision": ctx_prec,
        "context_recall": ctx_rec,
        "bert_precision": float(P[0]),
        "bert_recall": float(R[0]),
        "bert_f1": float(F1[0]),
        "num_chunks": len(chunks),
    }


# Carga de multiples dataset

def load_dataset(fname):
    qa_dir = get_QA_dir()
    path = os.path.join(qa_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe archivo: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Evaluacion Global

def summarize_metrics(title, metrics_list):
    print(f"\n[{title}]")
    keys = metrics_list[0].keys()
    for k in keys:
        vals = [m[k] for m in metrics_list]
        print(f"{k}: {sum(vals)/len(vals):.3f} (n={len(vals)})")


def evaluate_per_dataset(files, verbose=False):
    print("\n===== RESULTADOS POR DATASET =====")

    per_ds_results = {}

    for fname in files:
        print(f"\n Evaluando dataset {fname} ...")

        data = load_dataset(fname)
        local_metrics = []

        for item in data:
            _, m = evaluate_one_question(item["question"], item["answer"], verbose)
            local_metrics.append(m)

        per_ds_results[fname] = local_metrics
        summarize_metrics(fname, local_metrics)

    return per_ds_results


def evaluate_dataset_multi(files, verbose=False):

    # Evaluación per-dataset si se pidió
    if args.per_dataset:
        evaluate_per_dataset(files, verbose)

    # Evaluación global combinada
    all_items = []
    for f in files:
        all_items.extend(load_dataset(f))

    print(f"\nEvaluando {len(all_items)} preguntas combinadas de:")
    for f in files:
        print(" -", f)

    all_metrics = []
    for item in all_items:
        _, m = evaluate_one_question(item["question"], item["answer"], verbose)
        all_metrics.append(m)

    print("\n===== RESUMEN GLOBAL =====")
    summarize_metrics("GLOBAL", all_metrics)


# MAIN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluación unificada de múltiples QA-datasets.")
    parser.add_argument("qa_files", nargs="+", help="Archivos QA JSON")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--per_dataset", action="store_true", help="Mostrar métricas por dataset")
    args = parser.parse_args()

    evaluate_dataset_multi(args.qa_files, verbose=args.verbose)
