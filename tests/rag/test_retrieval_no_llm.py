"""Ensure retrieval works without an LLM provider configured."""

from pathlib import Path

import pymupdf
from core.config import Settings
from core.container import build_services


def _make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_retrieval_works_without_llm(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, "Retrieval augmented generation improves grounded answers.")

    settings = Settings(
        openai_api_key="",
        chroma_persist_dir=str(tmp_path / "chroma"),
        documents_dir=str(tmp_path / "documents"),
        chunk_size=200,
        chunk_overlap=20,
    )
    services = build_services(settings)

    from backend.services.document_service import DocumentService

    ds = DocumentService(settings, services.vector_store)
    ds.ingest_path(pdf_path)

    results = services.retriever.retrieve("grounded answers", top_k=3)
    assert len(results) >= 1
    assert results[0].chunk_id