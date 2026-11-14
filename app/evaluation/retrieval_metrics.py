"""Standard ranked-retrieval metrics with binary relevance."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant) if relevant else 0.0


def hit_rate_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: Sequence[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked, 1):
        if cid in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], 1) if cid in relevant)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def per_question(ranked: Sequence[str], relevant: set[str], k: int) -> dict[str, float]:
    return {f"recall@{k}": recall_at_k(ranked, relevant, k), "mrr": reciprocal_rank(ranked, relevant),
            f"ndcg@{k}": ndcg_at_k(ranked, relevant, k), f"hit_rate@{k}": hit_rate_at_k(ranked, relevant, k)}


def mean(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {**{k: sum(r[k] for r in rows) / len(rows) for k in keys}, "n": len(rows)}
