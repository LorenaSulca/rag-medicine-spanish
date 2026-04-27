from rag.schemas import ValidationResult


def validate_sentences(
    answer: str,
    chunks: list,
    threshold: float = 0.22,
    partial_threshold: float = 0.01,
    invalid_threshold: float = 0.5,
):
    """
    Stub inicial: no valida nada todavía.
    """

    return ValidationResult(
        usr=0.0,
        unsupported_sentences=[],
        sentence_results=[],
        is_valid=True,
        decision="answered",
    )