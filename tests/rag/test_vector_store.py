"""Tests for vector store ingestion and deletion."""

from pathlib import Path

import pymupdf
from core.config import Settings

from rag.embeddings import EmbeddingModel
from rag.ingestion import ingest_pdf
from rag.vector_store import VectorStore


def _make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_ingest_skips_already_indexed_document(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, "Retrieval augmented generation improves grounded answers.")

    settings = Settings(chunk_size=100, chunk_overlap=10)
    embedding_model = EmbeddingModel("all-MiniLM-L6-v2")
    store = VectorStore(tmp_path / "chroma", embedding_model)

    first = ingest_pdf(pdf_path, settings, store)
    assert first.skipped is False
    assert first.chunks > 0

    second = ingest_pdf(pdf_path, settings, store)
    assert second.skipped is True
    assert second.chunks == 0

    docs = store.list_documents()
    assert len(docs) == 1

    store.delete_document(first.document_id)
    assert store.list_documents() == []