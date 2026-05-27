import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer


class MedCPTEmbedder:
    """
    Embedder biomédico basado en MedCPT.

    IMPORTANTE:
    MedCPT usa modelos distintos para:
    - queries
    - documentos

    Esto es intencional y forma parte de su diseño retrieval-oriented.
    """

    def __init__(
        self,
        query_model_name: str = "ncbi/MedCPT-Query-Encoder",
        document_model_name: str = "ncbi/MedCPT-Article-Encoder",
        device: str | None = None,
    ):
        self.query_model_name = query_model_name
        self.document_model_name = document_model_name

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        # Query encoder
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            self.query_model_name
        )

        self.query_model = AutoModel.from_pretrained(
            self.query_model_name
        )

        self.query_model.to(self.device)
        self.query_model.eval()

        # Document encoder
        self.document_tokenizer = AutoTokenizer.from_pretrained(
            self.document_model_name
        )

        self.document_model = AutoModel.from_pretrained(
            self.document_model_name
        )

        self.document_model.to(self.device)
        self.document_model.eval()

    @property
    def name(self) -> str:
        return "medcpt"

    @property
    def dimension(self) -> int:
        return 768

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(
            text=text,
            tokenizer=self.query_tokenizer,
            model=self.query_model,
        )

    def embed_document(self, text: str) -> np.ndarray:
        return self._embed(
            text=text,
            tokenizer=self.document_tokenizer,
            model=self.document_model,
        )

    def _mean_pooling(
        self,
        model_output,
        attention_mask,
    ):
        token_embeddings = model_output.last_hidden_state

        input_mask_expanded = (
            attention_mask.unsqueeze(-1)
            .expand(token_embeddings.size())
            .float()
        )

        return torch.sum(
            token_embeddings * input_mask_expanded,
            dim=1,
        ) / torch.clamp(
            input_mask_expanded.sum(dim=1),
            min=1e-9,
        )

    def _embed(
        self,
        text: str,
        tokenizer,
        model,
    ) -> np.ndarray:

        encoded_input = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        encoded_input = {
            k: v.to(self.device)
            for k, v in encoded_input.items()
        }

        with torch.no_grad():
            model_output = model(**encoded_input)

        sentence_embeddings = self._mean_pooling(
            model_output,
            encoded_input["attention_mask"],
        )

        sentence_embeddings = F.normalize(
            sentence_embeddings,
            p=2,
            dim=1,
        )

        return (
            sentence_embeddings[0]
            .cpu()
            .numpy()
            .astype("float32")
        )