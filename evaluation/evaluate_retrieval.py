"""Reproducible retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from core.config import get_settings
from core.container import build_services

from backend.services.document_service import DocumentService
from evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


def main() -> None:
    settings = get_settings()
    services = build_services(settings)
    doc_service = DocumentService(settings, services.vector_store)

    sample_path = Path(settings.documents_dir) / "sample_research.pdf"
    if sample_path.exists():
        doc_service.ingest_path(sample_path)

    dataset = json.loads(Path("evaluation/datasets/retrieval_eval.json").read_text(encoding="utf-8"))

    k = settings.retrieval_top_k
    retrieved_rankings: list[list[str]] = []
    relevant_sets: list[set[str]] = []
    per_question: list[dict] = []

    for item in dataset:
        question = item["question"]
        chunks = services.retriever.retrieve(question, top_k=k)
        retrieved_ids = [f"{c.metadata.get('source')}:{c.metadata.get('page')}" for c in chunks]
        relevant = {f"sample_research.pdf:{page}" for page in item["relevant_pages"]}
        retrieved_rankings.append(retrieved_ids)
        relevant_sets.append(relevant)

        per_question.append(
            {
                "id": item["id"],
                "question": question,
                "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
                "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
                "reciprocal_rank": (1.0 / (retrieved_ids.index(next((x for x in retrieved_ids if x in relevant), "")) + 1)) if any(x in relevant for x in retrieved_ids) else 0.0,
                "retrieved": retrieved_ids,
            }
        )

    summary = {
        "top_k": k,
        "mean_precision_at_k": sum(q["precision_at_k"] for q in per_question) / len(per_question),
        "mean_recall_at_k": sum(q["recall_at_k"] for q in per_question) / len(per_question),
        "mrr": mean_reciprocal_rank(retrieved_rankings, relevant_sets),
        "per_question": per_question,
        "status": "completed",
    }

    output = Path("evaluation/results/retrieval_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_question"}, indent=2))


if __name__ == "__main__":
    main()