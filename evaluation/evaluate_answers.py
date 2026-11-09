"""Answer evaluation (latency, tokens, citation presence)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from core.config import get_settings
from core.container import build_services

from backend.services.document_service import DocumentService


def main() -> None:
    settings = get_settings()
    services = build_services(settings)
    doc_service = DocumentService(settings, services.vector_store)

    sample_path = Path(settings.documents_dir) / "sample_research.pdf"
    if sample_path.exists():
        doc_service.ingest_path(sample_path)

    dataset = json.loads(Path("evaluation/datasets/retrieval_eval.json").read_text(encoding="utf-8"))
    records = []
    has_key = bool(settings.openai_api_key)
    for item in dataset:
        if has_key:
            result = services.rag_pipeline.answer(item["question"], top_k=settings.retrieval_top_k)
            cited = re.findall(r"\[[^\]]+:\s*\d+\]", result.answer)
            record = {
                "id": item["id"],
                "question": item["question"],
                "answer": result.answer,
                "sufficient_context": result.sufficient_context,
                "num_sources": len(result.sources),
                "num_citations": len(cited),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "latency_seconds": result.latency_seconds,
                "faithfulness": "pending (requires LLM-as-judge or human review)",
                "answer_relevance": "pending (requires LLM-as-judge or human review)",
                "citation_correctness": "pending (requires manual verification)",
            }
        else:
            record = {
                "id": item["id"],
                "question": item["question"],
                "answer": "pending (OPENAI_API_KEY not configured)",
                "sufficient_context": None,
                "num_sources": 0,
                "num_citations": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
                "faithfulness": "pending",
                "answer_relevance": "pending",
                "citation_correctness": "pending",
            }
        records.append(record)

    output = Path("evaluation/results/answer_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "completed", "records": records}, indent=2), encoding="utf-8")
    print(f"Evaluated {len(records)} answers.")


if __name__ == "__main__":
    main()