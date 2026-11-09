"""Tests for hybrid retrieval and RRF."""

from core.config import Settings

from rag.hybrid_retriever import HybridRetriever


class FakeVectorStore:
    def query(self, query_text, top_k=5, document_id=None):
        return [
            {"id": "a", "content": "chunk a", "metadata": {"source": "d.pdf", "page": 1}, "score": 0.9},
            {"id": "b", "content": "chunk b", "metadata": {"source": "d.pdf", "page": 2}, "score": 0.8},
        ][:top_k]


class FakeBM25Index:
    def search(self, query, top_k=5, document_id=None):
        return [
            {"id": "b", "content": "chunk b", "metadata": {"source": "d.pdf", "page": 2}, "score": 5.0},
            {"id": "c", "content": "chunk c", "metadata": {"source": "d.pdf", "page": 3}, "score": 4.0},
        ][:top_k]


def test_hybrid_rrf_consolidates_and_ranks() -> None:
    settings = Settings(rrf_k=60, vector_search_top_k=2, bm25_search_top_k=2, hybrid_final_top_k=3)
    retriever = HybridRetriever(FakeVectorStore(), FakeBM25Index(), settings)
    results = retriever.retrieve("test")
    ids = [r.chunk_id for r in results]
    assert "b" in ids  # appears in both
    # b should rank first due to appearing in both lists
    assert results[0].chunk_id == "b"
    b = next(r for r in results if r.chunk_id == "b")
    assert "vector" in b.sources and "bm25" in b.sources
    assert b.vector_score == 0.8
    assert b.bm25_score == 5.0