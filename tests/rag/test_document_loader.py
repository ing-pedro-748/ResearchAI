"""Tests for PDF document loading."""

from pathlib import Path

import pymupdf
import pytest

from rag.document_loader import DocumentLoader


def _make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_load_pdf_extracts_page_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, "Hello ResearchAI")

    loader = DocumentLoader()
    document_id, documents = loader.load_pdf(pdf_path)

    assert len(document_id) == 16
    assert len(documents) == 1
    assert "Hello ResearchAI" in documents[0].page_content
    assert documents[0].metadata["page"] == 1
    assert documents[0].metadata["source"] == "sample.pdf"
    assert documents[0].metadata["document_id"] == document_id


def test_load_pdf_missing_file_raises(tmp_path: Path) -> None:
    loader = DocumentLoader()
    with pytest.raises(FileNotFoundError):
        loader.load_pdf(tmp_path / "missing.pdf")