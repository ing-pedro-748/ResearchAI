"""PDF document loading using PyMuPDF."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
from langchain_core.documents import Document


class DocumentLoader:
    """Load PDF files into LangChain documents with page-level metadata."""

    def load_pdf(self, path: str | Path) -> tuple[str, list[Document]]:
        """Return (document_id, documents) for a PDF file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        file_bytes = path.read_bytes()
        document_id = hashlib.sha256(file_bytes).hexdigest()[:16]

        pdf = pymupdf.open(path)
        documents: list[Document] = []
        try:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text()
                if not text or not text.strip():
                    continue
                metadata = {
                    "source": path.name,
                    "page": page_number,
                    "document_id": document_id,
                }
                title = pdf.metadata.get("title") if pdf.metadata else None
                if title:
                    metadata["title"] = title
                documents.append(Document(page_content=text, metadata=metadata))
        finally:
            pdf.close()

        if not documents:
            raise ValueError(f"No extractable text found in PDF: {path}")

        return document_id, documents