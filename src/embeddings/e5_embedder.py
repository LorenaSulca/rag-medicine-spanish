import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class E5Embedder:
    """
    Embedder multilingüe orientado a retrieval.

    Modelo sugerido:
    intfloat/multilingual-e5-large

    Importante:
    Los modelos E5 recomiendan prefijos:
    - query: para consultas
    - passage: para documentos
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        device: str | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def name(self) -> str:
        return "multilingual_e5"

    @property
    def dimension(self) -> int:
        return 1024

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(f"query: {text}")

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(f"passage: {text}")

    def _average_pool(
        self,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(),
            0.0,
        )

        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _embed(self, text: str) -> np.ndarray:
        batch = self.tokenizer(
            [text],
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        batch = {
            key: value.to(self.device)
            for key, value in batch.items()
        }

        with torch.no_grad():
            outputs = self.model(**batch)

        embeddings = self._average_pool(
            outputs.last_hidden_state,
            batch["attention_mask"],
        )

        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings[0].cpu().numpy().astype("float32")