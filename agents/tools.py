"""Research agent tools."""

from __future__ import annotations

import json
from typing import Any

from core.config import Settings
from core.llm_provider import LLMProvider
from langchain_core.tools import tool

from rag.retriever import Retriever
from rag.vector_store import VectorStore


class ResearchTools:
    """Container for tool implementations used by the research agent."""

    def __init__(
        self,
        retriever: Retriever,
        vector_store: VectorStore,
        llm_provider: LLMProvider,
        settings: Settings,
    ) -> None:
        self.retriever = retriever
        self.vector_store = vector_store
        self.llm_provider = llm_provider
        self.settings = settings

    def _context_from_chunks(self, chunks: list[dict[str, Any]] | list[Any]) -> str:
        parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            metadata = getattr(chunk, "metadata", None) or chunk.get("metadata", {})
            content = getattr(chunk, "content", None) or chunk.get("content", "")
            source = metadata.get("source", "unknown")
            page = metadata.get("page", "?")
            parts.append(f"[{index}] {source}, page {page}:\n{content}")
        return "\n\n".join(parts)

    def search_documents(self, query: str, top_k: int = 5) -> str:
        chunks = self.retriever.retrieve(query, top_k=top_k)
        if not chunks:
            return "No relevant documents found."
        return self._context_from_chunks(chunks)

    def summarize_document(self, document_id: str) -> str:
        chunks = self.vector_store.get_document_chunks(document_id)
        if not chunks:
            return "Document not found."
        context = self._context_from_chunks(chunks)
        from pathlib import Path

        template = Path("prompts/summarization.txt").read_text(encoding="utf-8")
        user_prompt = template.format(context=context)
        response = self.llm_provider.generate("You are a careful research assistant.", user_prompt)
        return response.text

    def compare_documents(self, document_ids: list[str]) -> str:
        all_chunks: list[dict[str, Any]] = []
        for doc_id in document_ids:
            for chunk in self.vector_store.get_document_chunks(doc_id):
                chunk = dict(chunk)
                chunk.setdefault("metadata", {})["document_id"] = doc_id
                all_chunks.append(chunk)
        if not all_chunks:
            return "No documents found."
        context = self._context_from_chunks(all_chunks)
        from pathlib import Path

        template = Path("prompts/comparison.txt").read_text(encoding="utf-8")
        user_prompt = template.format(context=context)
        response = self.llm_provider.generate("You are a careful research assistant.", user_prompt)
        return response.text

    def extract_methodology(self, document_id: str) -> str:
        chunks = self.vector_store.get_document_chunks(document_id)
        if not chunks:
            return "Document not found."
        context = self._context_from_chunks(chunks)
        from pathlib import Path

        template = Path("prompts/methodology_extraction.txt").read_text(encoding="utf-8")
        user_prompt = template.format(context=context)
        response = self.llm_provider.generate("You are a careful research assistant.", user_prompt)
        return response.text

    def generate_research_report(self, query: str) -> str:
        chunks = self.retriever.retrieve(query, top_k=self.settings.retrieval_top_k)
        if not chunks:
            return "Insufficient context to generate a report."
        context = self._context_from_chunks(
            [{"content": c.content, "metadata": c.metadata} for c in chunks]
        )
        from pathlib import Path

        template = Path("prompts/research_report.txt").read_text(encoding="utf-8")
        user_prompt = template.format(context=context, question=query)
        response = self.llm_provider.generate("You are a careful research assistant.", user_prompt)
        return response.text


def make_tools(research_tools: ResearchTools) -> list[Any]:
    @tool
    def search_documents(query: str, top_k: int = 5) -> str:
        """Search indexed documents for relevant passages."""
        return research_tools.search_documents(query, top_k)

    @tool
    def summarize_document(document_id: str) -> str:
        """Generate a structured summary of a document."""
        return research_tools.summarize_document(document_id)

    @tool
    def compare_documents(document_ids: list[str]) -> str:
        """Compare multiple documents."""
        return research_tools.compare_documents(document_ids)

    @tool
    def extract_methodology(document_id: str) -> str:
        """Extract models, datasets, methods, and metrics from a document."""
        return research_tools.extract_methodology(document_id)

    @tool
    def generate_research_report(query: str) -> str:
        """Generate a structured research report from retrieved information."""
        return research_tools.generate_research_report(query)

    return [search_documents, summarize_document, compare_documents, extract_methodology, generate_research_report]


def parse_tool_selection(text: str) -> tuple[str, dict[str, Any]]:
    """Parse LLM JSON tool selection; fall back to search_documents."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("no json")
        payload = json.loads(text[start : end + 1])
        tool_name = payload.get("tool", "search_documents")
        args = payload.get("arguments", {}) or {}
        return tool_name, args
    except (json.JSONDecodeError, ValueError):
        return "search_documents", {}