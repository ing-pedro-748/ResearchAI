"""ChromaDB vector store wrapper."""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document

from rag.embeddings import EmbeddingModel


class VectorStore:
    """Persistent ChromaDB-backed store for document chunks."""

    COLLECTION_NAME = "documents"

    def __init__(self, persist_dir: str | Path, embedding_model: EmbeddingModel) -> None:
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        chroma_settings = chromadb.Settings(anonymized_telemetry=False)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir), settings=chroma_settings)
        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def has_document(self, document_id: str) -> bool:
        result = self.collection.get(where={"document_id": document_id}, limit=1)
        return bool(result.get("ids"))

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        embeddings = self.embedding_model.embed_documents(texts)
        ids = [str(uuid.uuid4()) for _ in documents]
        metadatas: list[dict[str, Any]] = [dict(doc.metadata) for doc in documents]

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(documents)

    def delete_document(self, document_id: str) -> None:
        self.collection.delete(where={"document_id": document_id})

    def get_document_chunks(self, document_id: str) -> list[dict[str, Any]]:
        result = self.collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        chunks: list[dict[str, Any]] = []
        for text, metadata in zip(result.get("documents", []) or [], result.get("metadatas", []) or [], strict=False):
            chunks.append({"content": text, "metadata": metadata or {}})
        return chunks

    def get_all_chunks(self) -> list[dict[str, Any]]:
        result = self.collection.get(include=["documents", "metadatas"])
        chunks: list[dict[str, Any]] = []
        ids = result.get("ids", []) or []
        documents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=False):
            chunks.append(
                {
                    "id": chunk_id,
                    "content": text,
                    "metadata": metadata or {},
                }
            )
        return chunks

    def list_documents(self) -> list[dict[str, str]]:
        result = self.collection.get(include=["metadatas"])
        seen: dict[str, str] = {}
        for metadata in result.get("metadatas", []) or []:
            doc_id = metadata.get("document_id")
            if doc_id and doc_id not in seen:
                seen[doc_id] = metadata.get("source", "")
        return [{"document_id": k, "source": v} for k, v in seen.items()]

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            return []
        query_embedding = self.embedding_model.embed_query(query_text)
        where = {"document_id": document_id} if document_id else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        output: list[dict[str, Any]] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            score = 1.0 - float(distance)
            output.append(
                {
                    "id": chunk_id,
                    "content": text,
                    "metadata": metadata or {},
                    "score": score,
                }
            )
        return output