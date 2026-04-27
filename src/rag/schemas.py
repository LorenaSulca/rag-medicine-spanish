from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal


RAGStatus = Literal[
    "answered",
    "partial",
    "abstained",
    "invalidated",
    "no_context",
    "error",
]


@dataclass
class Chunk:
    text: str
    chunk_id: str | int = "?"
    document_id: str = "desconocido"
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        known = {
            "text",
            "chunk_id",
            "document_id",
            "score",
        }

        metadata = {k: v for k, v in data.items() if k not in known}

        return cls(
            text=data.get("text", ""),
            chunk_id=data.get("chunk_id", "?"),
            document_id=data.get("document_id", "desconocido"),
            score=float(data.get("score", 0.0)),
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        base = asdict(self)
        metadata = base.pop("metadata", {})
        return {**base, **metadata}


@dataclass
class RetrievalResult:
    chunks: List[Chunk]
    signals: Dict[str, List[str]] = field(default_factory=dict)
    medspaner_raw: Any = None

    @property
    def contexts(self) -> List[str]:
        return [chunk.text for chunk in self.chunks]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": [c.to_dict() for c in self.chunks],
            "contexts": self.contexts,
            "signals": self.signals,
            "medspaner_raw": self.medspaner_raw,
        }


@dataclass
class GenerationResult:
    answer: str
    raw_response: Any = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "raw_response": self.raw_response,
        }


@dataclass
class SentenceValidationItem:
    sentence: str
    supported: bool
    max_similarity: float
    best_chunk_id: str | int | None = None
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    usr: float = 0.0
    unsupported_sentences: List[str] = field(default_factory=list)
    sentence_results: List[SentenceValidationItem] = field(default_factory=list)
    citation_consistency: Optional[float] = None
    is_valid: bool = True
    decision: RAGStatus = "answered"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "usr": self.usr,
            "unsupported_sentences": self.unsupported_sentences,
            "sentence_results": [s.to_dict() for s in self.sentence_results],
            "citation_consistency": self.citation_consistency,
            "is_valid": self.is_valid,
            "decision": self.decision,
        }


@dataclass
class RAGResponse:
    question: str
    answer: str
    status: RAGStatus
    experiment: str
    contexts: List[str]
    chunks: List[Chunk]
    signals: Dict[str, List[str]] = field(default_factory=dict)
    medspaner_raw: Any = None
    validation: Optional[ValidationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "experiment": self.experiment,
            "contexts": self.contexts,
            "chunks": [c.to_dict() for c in self.chunks],
            "signals": self.signals,
            "medspaner_raw": self.medspaner_raw,
            "validation": self.validation.to_dict() if self.validation else None,
            "metadata": self.metadata,
        }