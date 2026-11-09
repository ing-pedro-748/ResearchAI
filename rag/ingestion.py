"""High-level PDF ingestion orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.config import Settings

from rag.chunking import chunk_documents
from rag.document_loader import DocumentLoader
from rag.vector_store import VectorStore


@dataclass
class IngestionResult:
    document_id: str
    source: str
    chunks: int
    skipped: bool


def ingest_pdf(
    path: str | Path,
    settings: Settings,
    vector_store: VectorStore,
    loader: DocumentLoader | None = None,
) -> IngestionResult:
    path = Path(path)
    loader = loader or DocumentLoader()
    document_id, documents = loader.load_pdf(path)

    if vector_store.has_document(document_id):
        return IngestionResult(
            document_id=document_id,
            source=path.name,
            chunks=0,
            skipped=True,
        )

    chunks = chunk_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    added = vector_store.add_documents(chunks)
    return IngestionResult(
        document_id=document_id,
        source=path.name,
        chunks=added,
        skipped=False,
    )