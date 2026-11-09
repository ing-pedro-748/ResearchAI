"""End-to-end Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from core.llm_provider import LLMProvider

from rag.retriever import RetrievedChunk, Retriever


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_seconds: float = 0.0
    sufficient_context: bool = True


class RAGPipeline:
    """Combine retrieval and LLM generation into a grounded answer."""

    def __init__(
        self,
        retriever: Retriever,
        llm_provider: LLMProvider,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.prompt_path = Path(prompt_path) if prompt_path else (
            Path(__file__).resolve().parents[1] / "prompts" / "question_answering.txt"
        )
        self._template = self.prompt_path.read_text(encoding="utf-8")

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        parts: list[str] = []
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            parts.append(f"[{index}] {source}, page {page}:\n{chunk.content}")
        return "\n\n".join(parts)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        document_id: str | None = None,
    ) -> RAGAnswer:
        start = time.perf_counter()
        chunks = self.retriever.retrieve(question, top_k=top_k, document_id=document_id)

        if not chunks:
            return RAGAnswer(
                answer="The available documents do not contain enough information to answer this question.",
                sources=[],
                sufficient_context=False,
                latency_seconds=time.perf_counter() - start,
            )

        context = self._build_context(chunks)
        user_prompt = self._template.format(context=context, question=question)
        system_prompt = (
            "You are a careful research assistant. Ground every claim in the provided context."
        )

        response = self.llm_provider.generate(system_prompt, user_prompt)
        return RAGAnswer(
            answer=response.text,
            sources=chunks,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            latency_seconds=time.perf_counter() - start,
            sufficient_context=True,
        )