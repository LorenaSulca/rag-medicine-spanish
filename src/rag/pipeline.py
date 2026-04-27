from __future__ import annotations

from typing import Any, Dict, List

from rag.schemas import (
    Chunk,
    RetrievalResult,
    GenerationResult,
    ValidationResult,
    RAGResponse,
)

from retrieval.hybrid_retriever import retrieve_hybrid
from retrieval.base_retriever import retrieve_base
from generation.generator import generate_answer
from validation.sentence_validator import validate_sentences


NO_CONTEXT_MESSAGE = (
    "No se encontraron fragmentos relevantes en la base de prospectos para responder la pregunta."
)

ABSTENTION_MESSAGE = (
    "No se puede responder con la información disponible en el contexto proporcionado."
)

ERROR_MESSAGE = (
    "Ocurrió un error durante el procesamiento de la consulta."
)


class RAGPipeline:
    def __init__(
        self,
        llm_client: Any,
        config: Dict[str, Any],
        experiment_name: str = "baseline",
    ):
        self.llm_client = llm_client
        self.config = config
        self.experiment_name = experiment_name

    def run(self, question: str) -> RAGResponse:
        try:
            retrieval = self._retrieve(question)

            if not retrieval.chunks:
                return RAGResponse(
                    question=question,
                    answer=NO_CONTEXT_MESSAGE,
                    status="no_context",
                    experiment=self.experiment_name,
                    contexts=[],
                    chunks=[],
                    signals=retrieval.signals,
                    medspaner_raw=retrieval.medspaner_raw,
                )

            generation = self._generate(question, retrieval)
            answer = generation.answer.strip()

            if answer == ABSTENTION_MESSAGE:
                return RAGResponse(
                    question=question,
                    answer=ABSTENTION_MESSAGE,
                    status="abstained",
                    experiment=self.experiment_name,
                    contexts=retrieval.contexts,
                    chunks=retrieval.chunks,
                    signals=retrieval.signals,
                    medspaner_raw=retrieval.medspaner_raw,
                )

            if self.config.get("refine_generation", False):
                answer = self._refine(question, answer, retrieval)

            validation = None
            status = "answered"

            if self.config.get("sentence_validation", False):
                validation = self._validate_sentences(answer, retrieval)
                status = validation.decision

                if status == "invalidated":
                    answer = (
                        "La respuesta generada no pudo ser validada contra el contexto disponible. "
                        "No se puede responder con suficiente fiabilidad."
                    )

            return RAGResponse(
                question=question,
                answer=answer,
                status=status,
                experiment=self.experiment_name,
                contexts=retrieval.contexts,
                chunks=retrieval.chunks,
                signals=retrieval.signals,
                medspaner_raw=retrieval.medspaner_raw,
                validation=validation,
                metadata={
                    "config": self.config,
                },
            )

        except Exception as exc:
            return RAGResponse(
                question=question,
                answer=ERROR_MESSAGE,
                status="error",
                experiment=self.experiment_name,
                contexts=[],
                chunks=[],
                metadata={
                    "error": str(exc),
                    "config": self.config,
                },
            )

    def _retrieve(self, question: str) -> RetrievalResult:
        if self.config.get("hybrid_retrieval", False):
            raw_chunks, signals, medspaner_raw = retrieve_hybrid(
                question,
                dynamic_k=self.config.get("dynamic_k", False),
            )
        else:
            raw_chunks, signals, medspaner_raw = retrieve_base(question)

        chunks = [Chunk.from_dict(ch) for ch in raw_chunks]

        return RetrievalResult(
            chunks=chunks,
            signals=signals,
            medspaner_raw=medspaner_raw,
        )

    def _generate(
        self,
        question: str,
        retrieval: RetrievalResult,
    ) -> GenerationResult:
        answer = generate_answer(
            llm_client=self.llm_client,
            question=question,
            chunks=[c.to_dict() for c in retrieval.chunks],
            signals=retrieval.signals,
            citation_prompt=self.config.get("citation_prompt", False),
        )

        return GenerationResult(answer=answer)

    def _refine(
        self,
        question: str,
        answer: str,
        retrieval: RetrievalResult,
    ) -> str:
        raise NotImplementedError("refine_generation aún no está implementado.")

    def _validate_sentences(
        self,
        answer: str,
        retrieval: RetrievalResult,
    ) -> ValidationResult:
        return validate_sentences(
            answer=answer,
            chunks=[c.to_dict() for c in retrieval.chunks],
            threshold=self.config.get("sentence_similarity_threshold", 0.22),
            partial_threshold=self.config.get("usr_partial_threshold", 0.01),
            invalid_threshold=self.config.get("usr_invalid_threshold", 0.50),
        )