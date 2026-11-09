"""BM25 lexical index synchronized with ChromaDB chunks."""

from __future__ import annotations

import re
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize text while preserving technical terms and identifiers."""
    if not text:
        return []
    return [token.lower() for token in re.findall(r"\w+(?:[-_]\w+)*", text, flags=re.UNICODE)]


class BM25Index:
    """In-memory BM25 index rebuilt from persisted ChromaDB chunks."""

    def __init__(self) -> None:
        self.ids: list[str] = []
        self.texts: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None

    def build(self, chunks: list[dict[str, Any]]) -> None:
        self.ids = [str(chunk["id"]) for chunk in chunks]
        self.texts = [str(chunk.get("content", "")) for chunk in chunks]
        self.metadatas = [dict(chunk.get("metadata", {}) or {}) for chunk in chunks]
        tokenized_corpus = [tokenize(text) for text in self.texts]
        if tokenized_corpus and any(tokens for tokens in tokenized_corpus):
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None

    def search(self, query: str, top_k: int = 10, document_id: str | None = None) -> list[dict[str, Any]]:
        if self._bm25 is None or not self.ids:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[dict[str, Any]] = []
        for index in ranked_indices:
            metadata = self.metadatas[index]
            if document_id is not None and metadata.get("document_id") != document_id:
                continue
            results.append(
                {
                    "id": self.ids[index],
                    "content": self.texts[index],
                    "metadata": metadata,
                    "score": float(scores[index]),
                }
            )
            if len(results) >= top_k:
                break
        return results

    def size(self) -> int:
        return len(self.ids)