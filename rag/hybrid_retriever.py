"""Hybrid retrieval combining vector search and BM25 via RRF."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from core.config import Settings

from rag.bm25_index import BM25Index
from rag.vector_store import VectorStore


@dataclass
class HybridResult:
    chunk_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    rrf_score: float = 0.0
    vector_score: float | None = None
    bm25_score: float | None = None
    sources: list[str] = field(default_factory=list)


class HybridRetriever:
    """Combine ChromaDB vector search and BM25 using Reciprocal Rank Fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        settings: Settings,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.settings = settings

    def _vector_search(self, query: str, top_k: int, document_id: str | None) -> list[dict[str, Any]]:
        return self.vector_store.query(query_text=query, top_k=top_k, document_id=document_id)

    def _bm25_search(self, query: str, top_k: int, document_id: str | None) -> list[dict[str, Any]]:
        return self.bm25_index.search(query=query, top_k=top_k, document_id=document_id)

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        vector_top_k: int | None = None,
        bm25_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> list[HybridResult]:
        if not query or not query.strip():
            return []

        v_top = vector_top_k if vector_top_k is not None else self.settings.vector_search_top_k
        b_top = bm25_top_k if bm25_top_k is not None else self.settings.bm25_search_top_k
        final_top = top_k if top_k is not None else self.settings.hybrid_final_top_k
        k = rrf_k if rrf_k is not None else self.settings.rrf_k

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_vector = executor.submit(self._vector_search, query, v_top, document_id)
            future_bm25 = executor.submit(self._bm25_search, query, b_top, document_id)
            vector_results = future_vector.result()
            bm25_results = future_bm25.result()

        fused: dict[str, HybridResult] = {}

        for rank, result in enumerate(vector_results, start=1):
            chunk_id = result["id"]
            if chunk_id not in fused:
                fused[chunk_id] = HybridResult(
                    chunk_id=chunk_id,
                    content=result["content"],
                    metadata=result.get("metadata", {}) or {},
                )
            fused[chunk_id].rrf_score += 1.0 / (k + rank)
            fused[chunk_id].vector_score = result.get("score")
            if "vector" not in fused[chunk_id].sources:
                fused[chunk_id].sources.append("vector")

        for rank, result in enumerate(bm25_results, start=1):
            chunk_id = result["id"]
            if chunk_id not in fused:
                fused[chunk_id] = HybridResult(
                    chunk_id=chunk_id,
                    content=result["content"],
                    metadata=result.get("metadata", {}) or {},
                )
            fused[chunk_id].rrf_score += 1.0 / (k + rank)
            fused[chunk_id].bm25_score = result.get("score")
            if "bm25" not in fused[chunk_id].sources:
                fused[chunk_id].sources.append("bm25")

        ordered = sorted(fused.values(), key=lambda item: item.rrf_score, reverse=True)
        return ordered[:final_top]