# RAG Farmacéutico en Español
## Ejecución del Pipeline

RAG (Retrieval-Augmented Generation) para responder preguntas sobre prospectos farmacéuticos en español.

1. Indexación (offline)  
2. Consulta (online)  
3. Evaluación  

---

# Requisitos previos

## Entorno Python

- Python 3.10+ (pipeline principal)  
- Python 3.7 (para MEDSPANER, en entorno separado)  

## Variables de entorno
Las variables por defecto consideran a medspaner inciado en el mismo directorio

- OPENAI_API_KEY  
- OLD_PYTHON_PATH
- MEDSPANER_SCRIPT
- MEDSPANER_CONFIG
- DATA_DIR
- QA_DIR

---

# Fase 1: Indexación (Offline)

Este proceso se ejecuta una sola vez por documento.

## Paso 1: Extraer texto desde PDF en carpeta prospects

Comando:
python pdf_text_extractor.py prospectoMedicamento.pdf medicamento.txt

Entrada:
- Archivo PDF del prospecto  

Salida:
- Archivo `.txt` limpio  

---

## Paso 2: Ejecutar MEDSPANER

Comando:
python medspaner_bridge.py medicamento.txt medicamento.json

Entrada:
- Texto limpio  

Salida:
- JSON con entidades biomédicas  

Notas:
- Este script usa un entorno Python 3.7 separado  
- No utiliza OpenAI  

---

## Paso 3: Generar chunks con información semántica

Comando:
python chunking_medspaner.py medicamento.txt medicamento.json medicamento_chunks.json

Entrada:
- Texto original  
- Entidades MEDSPANER  

Salida:
- JSON con chunks + metadata  

---

## Paso 4: Indexar en FAISS

Comando:
python indexar_faiss.py medicamento_chunks.json medicamento

Entrada:
- Chunks generados  

Salida:
- Índice FAISS  
- Metadata asociada  

Uso de OpenAI:
- Sí (embeddings)  
- Modelo: text-embedding-3-small  

---

# Fase 2: Consulta (Online)

## Ejecutar CLI de preguntas

Comando:
python cli_qa.py

Funcionamiento:
1. Se ingresa una pregunta por consola  
2. El sistema procesa y responde  

---

## Flujo interno de consulta

### 1. Procesamiento de la pregunta
- Se aplica MEDSPANER  
- Extrae entidades biomédicas  

Uso de OpenAI:
- No  

---

### 2. Generación de embedding
- Convierte la pregunta a vector  

Uso de OpenAI:
- Sí  
- Modelo: text-embedding-3-small  

---

### 3. Recuperación en FAISS
- Obtiene top-K chunks relevantes  

Uso de OpenAI:
- No  

---

### 4. Filtrado y reranking
- Ajusta ranking según entidades detectadas  

Tipo:
- Heurístico (no supervisado)  

Uso de OpenAI:
- No  

---

### 5. Generación de respuesta
- Se construye prompt con:
  - Pregunta  
  - Contexto recuperado  

Uso de OpenAI:
- Sí  
- Modelo: gpt-4o-mini  

---

# Fase 3: Evaluación

## Ejecutar evaluación completa

Comando:
python evaluation.py  

---

## Qué hace evaluation.py

1. Carga datasets QA:
   - medicamentoA_QA.json  
   - medicamentoB_QA.json  
   - medicamentoC_QA.json  

2. Para cada pregunta:
   - Ejecuta pipeline completo  
   - Genera respuesta  
   - Recupera contexto  

3. Calcula métricas automáticamente  

---

## Métricas utilizadas

### RAGAS (LLM-as-a-Judge)

Evalúa:
- Faithfulness  
- Answer Relevance  
- Context Precision  
- Context Recall  
- Context Relevance  

Uso de OpenAI:
- Sí (un LLM evalúa las respuestas)  

---

### BERTScore

Evalúa:
- Precision  
- Recall  
- F1  

Uso de OpenAI:
- No  

---

## Output esperado

Ejemplo:

[medicamento_QA.json]  
faithfulness: 0.656  
answer_relevance: 0.972  
context_relevance: 0.989  
context_precision: 0.770  
context_recall: 0.687  
bert_f1: 0.419  

===== RESUMEN GLOBAL =====  

faithfulness: 0.819  
answer_relevance: 0.987  
context_relevance: 0.993  
context_precision: 0.880  
context_recall: 0.803  
bert_f1: 0.461  

---

# Flujo completo resumido

1. python pdf_text_extractor.py archivo.pdf archivo.txt  
2. python medspaner_bridge.py archivo.txt archivo.json  
3. python chunking_medspaner.py archivo.txt archivo.json archivo_chunks.json  
4. python indexar_faiss.py archivo_chunks.json nombre_indice  
5. python cli_qa.py  
6. python evaluation.py  

---

# Uso de OpenAI en el sistema

- Embeddings → Sí  
- Generación de respuestas → Sí  
- Evaluación RAGAS → Sí  
- MEDSPANER → No  
- FAISS → No  
- BERTScore → No  

---

# Notas importantes

- El sistema depende de MEDSPANER para enriquecimiento semántico  
- El reranking actual es heurístico, no entrenado  
- El dataset QA es generado automáticamente mediante IA  
- No hay fine-tuning de modelos  

---

# Posibles mejoras futuras

- Incorporar rerankers supervisados  
- Usar embeddings biomédicos especializados  
- Integrar agentes para validación  
- Mejorar control de alucinaciones  