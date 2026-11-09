"""Tests for evaluation metrics."""

from evaluation.metrics import mean_reciprocal_rank, precision_at_k, recall_at_k


def test_precision_recall_mrr() -> None:
    retrieved = ["a", "b", "c"]
    relevant = {"b", "d"}
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mean_reciprocal_rank([retrieved], [relevant]) == 0.5