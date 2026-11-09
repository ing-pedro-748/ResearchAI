"""Tests for document chunking."""

import pytest
from langchain_core.documents import Document

from rag.chunking import chunk_documents


def test_chunk_documents_splits_and_preserves_metadata() -> None:
    doc = Document(page_content="A" * 2500, metadata={"source": "x.pdf", "page": 1})
    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=100)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "x.pdf"
        assert chunk.metadata["page"] == 1
        assert len(chunk.page_content) <= 1000


def test_chunk_documents_invalid_params() -> None:
    doc = Document(page_content="text", metadata={})
    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=100, chunk_overlap=100)