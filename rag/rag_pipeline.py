"""End-to-end Retrieval-Augmented Generation pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from core.llm_provider import LLMProvider
from core.tokenizer import count_tokens, truncate_text
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

    def _available_context_tokens(self, question: str) -> int:
        system_prompt = "You are a careful research assistant. Ground every claim in the provided context."
        system_tokens = count_tokens(system_prompt)
        question_tokens = count_tokens(question)
        template_tokens = count_tokens(self._template.replace("{context}", "").replace("{question}", ""))
        reserved = (
            self.llm_provider.settings.llm_reserved_answer_tokens
            if hasattr(self.llm_provider, "settings")
            else 512
        )
        safety = (
            self.llm_provider.settings.llm_prompt_safety_margin
            if hasattr(self.llm_provider, "settings")
            else 128
        )
        context_window = (
            self.llm_provider.settings.llm_context_window
            if hasattr(self.llm_provider, "settings")
            else 4096
        )
        available = context_window - system_tokens - question_tokens - template_tokens - reserved - safety
        return max(0, available)

    def _build_context(self, chunks: list[RetrievedChunk], question: str) -> str:
        available = self._available_context_tokens(question)
        parts: list[str] = []
        used = 0
        for index, chunk in enumerate(chunks, start=1):
            source = chunk.metadata.get("source", "unknown")
            page = chunk.metadata.get("page", "?")
            prefix = f"[{index}] {source}, page {page}:\n"
            prefix_tokens = count_tokens(prefix)
            remaining = available - used - prefix_tokens
            if remaining <= 0:
                break
            content = chunk.content
            content_tokens = count_tokens(content)
            if content_tokens > remaining:
                content = truncate_text(content, remaining)
            if not content.strip():
                continue
            parts.append(f"{prefix}{content}")
            used += count_tokens(f"{prefix}{content}") + 2
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

        context = self._build_context(chunks, question)
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