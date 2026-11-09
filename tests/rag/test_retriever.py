"""Tests for the retriever module."""

from pathlib import Path

import pymupdf
from core.config import Settings

from rag.embeddings import EmbeddingModel
from rag.ingestion import ingest_pdf
from rag.retriever import Retriever
from rag.vector_store import VectorStore


def _make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_retriever_returns_relevant_chunks(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, "Retrieval augmented generation improves grounded answers.")

    settings = Settings(chunk_size=200, chunk_overlap=20)
    model = EmbeddingModel("all-MiniLM-L6-v2")
    store = VectorStore(tmp_path / "chroma", model)
    ingest_pdf(pdf_path, settings, store)

    retriever = Retriever(store, top_k=3, score_threshold=0.0)
    results = retriever.retrieve("What improves grounded answers?")
    assert len(results) >= 1
    assert results[0].metadata["source"] == "paper.pdf"


def test_retriever_threshold_filters_results(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, "Retrieval augmented generation improves grounded answers.")

    settings = Settings(chunk_size=200, chunk_overlap=20)
    model = EmbeddingModel("all-MiniLM-L6-v2")
    store = VectorStore(tmp_path / "chroma", model)
    ingest_pdf(pdf_path, settings, store)

    retriever = Retriever(store, top_k=3, score_threshold=1.1)
    assert retriever.retrieve("unrelated query") == []