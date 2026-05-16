from typing import Literal, TypedDict, Optional, List


QACategory = Literal[
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

QADifficulty = Literal["easy", "medium", "hard"]

QAExpectedBehavior = Literal["answered", "partial", "abstained"]


class QAItem(TypedDict):
    id: str
    document_id: str
    category: QACategory
    difficulty: QADifficulty
    question: str
    ground_truth: str
    source_sections: List[str]
    expected_behavior: QAExpectedBehavior
    requires_numeric_grounding: bool
    requires_multi_hop: bool
    notes: Optional[str]