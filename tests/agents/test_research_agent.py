"""Tests for the research agent workflow."""

from pathlib import Path

import pymupdf
from core.config import Settings

from agents.research_agent import ResearchAgent
from rag.embeddings import EmbeddingModel
from rag.ingestion import ingest_pdf
from rag.vector_store import VectorStore


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str):
        from core.llm_provider import LLMResponse

        self.calls += 1
        if "Available tools" in user_prompt:
            return LLMResponse(text='{"tool": "search_documents", "arguments": {"query": "grounded", "top_k": 2}}')
        return LLMResponse(text="Final answer from agent.")


def _make_pdf(path: Path, text: str) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


def test_research_agent_runs_search_flow(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    _make_pdf(pdf_path, "Retrieval augmented generation improves grounded answers.")

    settings = Settings(chunk_size=200, chunk_overlap=20, max_agent_steps=8)
    model = EmbeddingModel("all-MiniLM-L6-v2")
    store = VectorStore(tmp_path / "chroma", model)
    ingest_pdf(pdf_path, settings, store)

    from rag.retriever import Retriever

    retriever = Retriever(store, top_k=3, score_threshold=0.0)
    agent = ResearchAgent(retriever, store, FakeLLM(), settings)

    answer = agent.final_answer("What improves grounded answers?")
    assert answer == "Final answer from agent."
    assert agent.run("What improves grounded answers?")["steps"] <= settings.max_agent_steps