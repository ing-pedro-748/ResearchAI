"""Reproducible retrieval evaluation: vector vs hybrid."""

from __future__ import annotations

import json
import time
from pathlib import Path

from core.config import get_settings
from core.container import build_services

from backend.services.document_service import DocumentService
from evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k
from rag.retriever import Retriever


def _evaluate(retriever, dataset, k, relevant_sets):
    retrieved_rankings: list[list[str]] = []
    per_question: list[dict] = []
    latencies: list[float] = []

    for item, relevant in zip(dataset, relevant_sets, strict=False):
        start = time.perf_counter()
        chunks = retriever.retrieve(item["question"], top_k=k)
        latency = time.perf_counter() - start
        latencies.append(latency)
        retrieved_ids = [f"{c.metadata.get('source')}:{c.metadata.get('page')}" for c in chunks]
        retrieved_rankings.append(retrieved_ids)
        per_question.append(
            {
                "id": item["id"],
                "question": item["question"],
                "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
                "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
                "reciprocal_rank": (
                    1.0 / (retrieved_ids.index(next((x for x in retrieved_ids if x in relevant), "")) + 1)
                    if any(x in relevant for x in retrieved_ids)
                    else 0.0
                ),
                "latency_seconds": latency,
                "retrieved": retrieved_ids,
            }
        )

    return {
        "top_k": k,
        "mean_precision_at_k": sum(q["precision_at_k"] for q in per_question) / len(per_question),
        "mean_recall_at_k": sum(q["recall_at_k"] for q in per_question) / len(per_question),
        "mrr": mean_reciprocal_rank(retrieved_rankings, relevant_sets),
        "mean_latency_seconds": sum(latencies) / len(latencies),
        "per_question": per_question,
    }


def main() -> None:
    settings = get_settings()
    services = build_services(settings)
    doc_service = DocumentService(settings, services.vector_store, services.bm25_index)

    sample_path = Path(settings.documents_dir) / "sample_research.pdf"
    if sample_path.exists():
        doc_service.ingest_path(sample_path)

    dataset = json.loads(Path("evaluation/datasets/retrieval_eval.json").read_text(encoding="utf-8"))
    k = settings.retrieval_top_k
    relevant_sets = [{f"sample_research.pdf:{page}" for page in item["relevant_pages"]} for item in dataset]

    vector_eval_retriever = Retriever(services.vector_store, top_k=k, score_threshold=0.0)
    vector_results = _evaluate(vector_eval_retriever, dataset, k, relevant_sets)
    hybrid_results = _evaluate(services.hybrid_retriever, dataset, k, relevant_sets)

    summary = {
        "vector_search": vector_results,
        "hybrid_search": hybrid_results,
        "status": "completed",
    }

    output = Path("evaluation/results/retrieval_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"vector": {kk: vv for kk, vv in vector_results.items() if kk != "per_question"}, "hybrid": {kk: vv for kk, vv in hybrid_results.items() if kk != "per_question"}}, indent=2))


if __name__ == "__main__":
    main()