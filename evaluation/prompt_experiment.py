"""Reproducible prompt configuration experiment."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from core.config import get_settings
from core.llm_provider import LLMProvider, OpenAIProvider

from rag.embeddings import EmbeddingModel
from rag.retriever import Retriever
from rag.vector_store import VectorStore

PROMPT_CONFIGS = [
    ("basic", Path("prompts/qa_basic.txt")),
    ("context_aware", Path("prompts/qa_context_aware.txt")),
    ("structured", Path("prompts/qa_structured.txt")),
]


def _build_context(chunks: list[Any]) -> str:
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[{index}] {source}, page {page}:\n{chunk.content}")
    return "\n\n".join(parts)


def run_prompt_experiment(
    retriever: Retriever,
    llm_provider: LLMProvider | None,
    dataset_path: str | Path = "evaluation/datasets/qa_prompts.json",
    output_path: str | Path = "evaluation/results/prompt_experiment.json",
    top_k: int = 5,
) -> dict[str, Any]:
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []

    for item in dataset:
        question = item["question"]
        document_id = item.get("document_id")
        chunks = retriever.retrieve(question, top_k=top_k, document_id=document_id)
        context = _build_context(chunks)

        for config_name, prompt_path in PROMPT_CONFIGS:
            template = prompt_path.read_text(encoding="utf-8")
            user_prompt = template.format(context=context, question=question)
            system_prompt = "You are a careful research assistant."

            record: dict[str, Any] = {
                "question_id": item["id"],
                "question": question,
                "config": config_name,
                "num_sources": len(chunks),
                "answer": None,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
                "status": "pending",
            }

            if llm_provider is not None:
                start = time.perf_counter()
                try:
                    response = llm_provider.generate(system_prompt, user_prompt)
                    record["answer"] = response.text
                    record["prompt_tokens"] = response.prompt_tokens
                    record["completion_tokens"] = response.completion_tokens
                    record["total_tokens"] = response.total_tokens
                    record["status"] = "completed"
                except Exception as exc:  # noqa: BLE001 - record failure
                    record["status"] = f"error: {exc}"
                record["latency_seconds"] = time.perf_counter() - start

            results.append(record)

    payload = {
        "status": "pending" if llm_provider is None else "completed",
        "note": "Metrics such as faithfulness and citation correctness require manual or LLM-as-judge evaluation. Results are pending until an API key is configured and the experiment is run.",
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    settings = get_settings()
    model = EmbeddingModel(settings.embedding_model)
    store = VectorStore(settings.chroma_persist_dir, model)
    retriever = Retriever(store, top_k=settings.retrieval_top_k, score_threshold=settings.retrieval_score_threshold)

    provider = OpenAIProvider(settings) if settings.openai_api_key else None
    payload = run_prompt_experiment(retriever, provider)
    print(f"Prompt experiment status: {payload['status']}")


if __name__ == "__main__":
    main()