"""Embedding model wrapper based on Sentence Transformers."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Thin wrapper around SentenceTransformer for document and query embeddings."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(text, show_progress_bar=False, convert_to_numpy=True)
        return vector.tolist()