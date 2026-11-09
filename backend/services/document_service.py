"""Document management service."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings

from rag.ingestion import IngestionResult, ingest_pdf
from rag.vector_store import VectorStore


class DocumentService:
    def __init__(self, settings: Settings, vector_store: VectorStore) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.documents_dir = Path(settings.documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[dict[str, str]]:
        return self.vector_store.list_documents()

    def ingest_path(self, path: Path) -> IngestionResult:

        settings = self.settings
        return ingest_pdf(path, settings, self.vector_store)

    def ingest_upload(self, filename: str, content: bytes) -> IngestionResult:
        destination = self.documents_dir / filename
        destination.write_bytes(content)
        return self.ingest_path(destination)

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)