"""LangGraph state definition for the research agent."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    request: str
    messages: list[dict[str, Any]]
    tool_name: str | None
    tool_args: dict[str, Any]
    tool_result: str | None
    final_answer: str | None
    steps: int
    error: str | None
    sufficient_context: bool