import numpy as np
from openai import OpenAI

from retrieval.utils_env import get_openai_api_key


class OpenAIEmbedder:
    """
    Embedder generalista basado en OpenAI.

    Se usa como baseline semántico actual del sistema.
    """

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
    ):
        self.model_name = model_name
        self.client = OpenAI(api_key=get_openai_api_key())

    @property
    def name(self) -> str:
        return "openai"

    @property
    def dimension(self) -> int | None:
        return None

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(text)

    def _embed(self, text: str) -> np.ndarray:
        response = self.client.embeddings.create(
            model=self.model_name,
            input=text,
        )

        return np.array(
            response.data[0].embedding,
            dtype="float32",
        )