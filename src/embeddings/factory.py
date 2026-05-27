from embeddings.openai_embedder import OpenAIEmbedder
from embeddings.e5_embedder import E5Embedder
from embeddings.medcpt_embedder import MedCPTEmbedder


SUPPORTED_EMBEDDERS = {
    "openai": OpenAIEmbedder,
    "multilingual_e5": E5Embedder,
    "e5": E5Embedder,
    "medcpt": MedCPTEmbedder,
}


def get_embedder(embedding_model: str = "openai"):
    """
    Factory central para seleccionar el modelo de embeddings.

    Valores soportados:
    - openai
    - multilingual_e5
    - e5
    - medcpt
    """

    key = embedding_model.lower().strip()

    if key not in SUPPORTED_EMBEDDERS:
        raise ValueError(
            f"Embedding model no soportado: {embedding_model}. "
            f"Opciones: {list(SUPPORTED_EMBEDDERS.keys())}"
        )

    return SUPPORTED_EMBEDDERS[key]()