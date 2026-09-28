"""Ranking metrics for the retrieval side of a trajectory.

Relevance is binary: a retrieved passage is relevant when its title appears in
the dataset's ``supporting_facts``. Metrics are computed over the merged,
rank-ordered candidate pool produced by all searches in one episode.
"""
from __future__ import annotations

from typing import Iterable, List, Sequence, Set


def gold_titles(supporting_facts) -> Set[str]:
    if not supporting_facts:
        return set()
    titles = set()
    for fact in supporting_facts:
        if isinstance(fact, dict):
            title = str(fact.get("title") or fact.get("doc_title") or "").strip().lower()
        else:
            title = str(fact).strip().lower()
        if title:
            titles.add(title)
    return titles


def _binary_relevance(ranked: Sequence[str], gold: Set[str]) -> List[int]:
    return [1 if str(t).strip().lower() in gold else 0 for t in ranked]


def recall_at_k(ranked: Sequence[str], gold: Set[str], k: int = 10) -> float:
    if not gold:
        return float("nan")
    hits = sum(_binary_relevance(list(ranked)[:k], gold))
    return hits / len(gold)


def mrr_at_k(ranked: Sequence[str], gold: Set[str], k: int = 10) -> float:
    if not gold:
        return float("nan")
    for rank, title in enumerate(list(ranked)[:k], start=1):
        if str(title).strip().lower() in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked: Sequence[str], gold: Set[str], k: int = 10) -> float:
    if not gold:
        return float("nan")
    rel = _binary_relevance(list(ranked)[:k], gold)
    dcg = sum(r / math_log2(i + 2) for i, r in enumerate(rel))
    ideal = [1] * min(len(gold), k)
    idcg = sum(r / math_log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def math_log2(value: float) -> float:
    import math

    return math.log2(value)


def merge_rankings(per_search: Iterable[Sequence[str]], scores: Iterable[Sequence[float]] = None) -> List[str]:
    """Merge per-search candidate lists, keeping the best (lowest) rank per title."""
    best: dict = {}
    for ranking in per_search:
        for rank, title in enumerate(ranking, start=1):
            key = str(title).strip().lower()
            if key and (key not in best or rank < best[key][0]):
                best[key] = (rank, str(title))
    return [title for _, title in sorted(best.values(), key=lambda x: x[0])]
