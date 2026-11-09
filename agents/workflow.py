"""LangGraph workflow for the research agent."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import Settings
from core.llm_provider import LLMProvider
from langgraph.graph import END, START, StateGraph

from agents.state import AgentState
from agents.tools import ResearchTools, parse_tool_selection


def build_workflow(
    research_tools: ResearchTools,
    llm_provider: LLMProvider,
    settings: Settings,
) -> Any:
    tool_map = {
        "search_documents": research_tools.search_documents,
        "summarize_document": research_tools.summarize_document,
        "compare_documents": research_tools.compare_documents,
        "extract_methodology": research_tools.extract_methodology,
        "generate_research_report": research_tools.generate_research_report,
    }

    def select_tool(state: AgentState) -> AgentState:
        steps = state.get("steps", 0)
        if steps >= settings.max_agent_steps:
            return {"error": "Maximum agent steps reached.", "steps": steps}

        request = state["request"]
        template = Path("prompts/agent_tool_selection.txt").read_text(encoding="utf-8")
        user_prompt = template.replace("{request}", request)
        response = llm_provider.generate("You are a research agent.", user_prompt)
        tool_name, tool_args = parse_tool_selection(response.text)
        if tool_name not in tool_map:
            tool_name = "search_documents"
            tool_args = {}
        return {
            "tool_name": tool_name,
            "tool_args": tool_args,
            "steps": steps + 1,
            "messages": state.get("messages", []) + [{"role": "assistant", "content": response.text}],
        }

    def execute_tool(state: AgentState) -> AgentState:
        if state.get("error"):
            return {}
        steps = state.get("steps", 0)
        if steps >= settings.max_agent_steps:
            return {"error": "Maximum agent steps reached.", "steps": steps}

        tool_name = state.get("tool_name") or "search_documents"
        tool_args = state.get("tool_args") or {}
        tool_fn = tool_map.get(tool_name)
        if tool_fn is None:
            return {"error": f"Unknown tool: {tool_name}", "steps": steps + 1}

        try:
            result = tool_fn(**tool_args)
        except Exception as exc:  # noqa: BLE001 - tool failure handling
            return {"error": f"Tool failed: {exc}", "tool_result": None, "steps": steps + 1}

        return {"tool_result": str(result), "steps": steps + 1}

    def finalize(state: AgentState) -> AgentState:
        steps = state.get("steps", 0)
        if state.get("error"):
            return {"final_answer": state["error"], "steps": steps}

        request = state["request"]
        tool_result = state.get("tool_result") or ""
        template_path = Path("prompts/question_answering.txt")
        template = template_path.read_text(encoding="utf-8")
        user_prompt = template.format(context=tool_result, question=request)
        response = llm_provider.generate("You are a careful research assistant.", user_prompt)
        return {"final_answer": response.text, "steps": steps + 1}

    graph = StateGraph(AgentState)
    graph.add_node("select_tool", select_tool)
    graph.add_node("execute_tool", execute_tool)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "select_tool")
    graph.add_edge("select_tool", "execute_tool")
    graph.add_edge("execute_tool", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()