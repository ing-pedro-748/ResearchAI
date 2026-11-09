"""Document management service."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings

from rag.ingestion import IngestionResult, ingest_pdf
from rag.vector_store import VectorStore


class DocumentService:
    def __init__(self, settings: Settings, vector_store: VectorStore, bm25_index=None) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.documents_dir = Path(settings.documents_dir)
        self.documents_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> list[dict[str, str]]:
        return self.vector_store.list_documents()

    def ingest_path(self, path: Path) -> IngestionResult:
        result = ingest_pdf(path, self.settings, self.vector_store)
        if self.bm25_index is not None:
            self.bm25_index.build(self.vector_store.get_all_chunks())
        return result

    def ingest_upload(self, filename: str, content: bytes) -> IngestionResult:
        destination = self.documents_dir / filename
        destination.write_bytes(content)
        return self.ingest_path(destination)

    def delete_document(self, document_id: str) -> None:
        self.vector_store.delete_document(document_id)
        if self.bm25_index is not None:
            self.bm25_index.build(self.vector_store.get_all_chunks())