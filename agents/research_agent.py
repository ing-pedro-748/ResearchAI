"""High-level research agent interface."""

from __future__ import annotations

from typing import Any

from core.config import Settings
from core.llm_provider import LLMProvider

from agents.state import AgentState
from agents.tools import ResearchTools, make_tools
from agents.workflow import build_workflow
from rag.retriever import Retriever
from rag.vector_store import VectorStore


class ResearchAgent:
    """LangGraph-based research assistant."""

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
        self.research_tools = ResearchTools(retriever, vector_store, llm_provider, settings)
        self.tools = make_tools(self.research_tools)
        self.graph = build_workflow(self.research_tools, llm_provider, settings)

    def run(self, request: str) -> dict[str, Any]:
        initial_state: AgentState = {
            "request": request,
            "messages": [],
            "steps": 0,
        }
        result = self.graph.invoke(initial_state)
        return result

    def final_answer(self, request: str) -> str:
        result = self.run(request)
        return result.get("final_answer") or "No answer generated."