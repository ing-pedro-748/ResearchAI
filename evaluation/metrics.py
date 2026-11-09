"""Evaluation metrics."""

from __future__ import annotations


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for item in top if item in relevant)
    return hits / len(top)


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for item in top if item in relevant)
    return hits / len(relevant)


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for index, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def mean_reciprocal_rank(rankings: list[list[str]], relevant_sets: list[set[str]]) -> float:
    if not rankings:
        return 0.0
    ranks = [reciprocal_rank(r, rel) for r, rel in zip(rankings, relevant_sets, strict=False)]
    return sum(ranks) / len(ranks)


def citation_correctness(cited_sources: list[str], retrieved_sources: set[str]) -> float:
    if not cited_sources:
        return 0.0
    hits = sum(1 for src in cited_sources if src in retrieved_sources)
    return hits / len(cited_sources)