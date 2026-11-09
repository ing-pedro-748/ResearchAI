"""Tests for BM25 index."""

from rag.bm25_index import BM25Index, tokenize


def test_tokenize_preserves_technical_terms() -> None:
    tokens = tokenize("Artículo 49 IDC-P GG")
    assert "artículo" in tokens
    assert "49" in tokens
    assert "idc-p" in tokens
    assert "gg" in tokens


def test_bm25_search_returns_ranked_results() -> None:
    index = BM25Index()
    index.build(
        [
            {"id": "1", "content": "Retrieval augmented generation improves grounded answers.", "metadata": {}},
            {"id": "2", "content": "BM25 is a lexical ranking function.", "metadata": {}},
            {"id": "3", "content": "Neural networks optimize deep representations.", "metadata": {}},
            {"id": "4", "content": "Transformers handle long context windows.", "metadata": {}},
        ]
    )
    results = index.search("lexical ranking", top_k=2)
    assert results[0]["id"] == "2"