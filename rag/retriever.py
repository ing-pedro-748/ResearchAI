"""Retrieval module independent from LLM generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag.vector_store import VectorStore


@dataclass
class RetrievedChunk:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class Retriever:
    """Semantic retriever over the vector store."""

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.vector_store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        k = top_k if top_k is not None else self.top_k
        if not query or not query.strip():
            return []

        raw_results = self.vector_store.query(
            query_text=query,
            top_k=k,
            document_id=document_id,
        )

        chunks: list[RetrievedChunk] = []
        for result in raw_results:
            score = float(result.get("score", 0.0))
            if score < self.score_threshold:
                continue
            chunks.append(
                RetrievedChunk(
                    content=result.get("content", ""),
                    metadata=result.get("metadata", {}) or {},
                    score=score,
                )
            )
        return chunks