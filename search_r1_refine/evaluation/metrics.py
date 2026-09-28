"""Answer, retrieval and judge metrics."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from search_r1_refine.rl.reward import score_trajectory
from search_r1_refine.rl.text import normalize
from search_r1_refine.evaluation.rank import gold_titles, mrr_at_k, ndcg_at_k, recall_at_k

RANK_K = 10


def _gold_answers(row):
    values = row.get("answers")
    if values is None or values == []:
        values = row.get("answer", [])
    if isinstance(values, str):
        return [values]
    if isinstance(values, (tuple, list)):
        return [str(x) for x in values]
    return [str(values)] if values else []


def _prompt_of(row):
    prompt = row.get("prompt_text") or row.get("prompt")
    if isinstance(prompt, list):
        prompt = prompt[-1].get("content", "") if prompt else ""
    return prompt or row.get("question", "")


def last_answer(text: str) -> str:
    from search_r1_refine.rl.reward import blocks

    found = blocks(text, "answer")
    return found[-1] if found else ""


def exact_match(prediction: str, golds: Iterable[str]) -> float:
    pred = normalize(prediction)
    if not pred:
        return 0.0
    return 1.0 if any(pred == normalize(g) for g in golds) else 0.0


def _mean(values: List[float]) -> float:
    values = [v for v in values if v is not None and v == v]
    return sum(values) / len(values) if values else 0.0


def score_row(row: Dict[str, Any], rank_k: int = RANK_K) -> Dict[str, float]:
    text = row.get("text", "")
    golds = _gold_answers(row)
    parts = score_trajectory(text, golds, row.get("supporting_facts"))
    prediction = last_answer(text)
    out = {
        "em": exact_match(prediction, golds),
        "answer_f1": float(parts.get("answer", 0.0)),
        "evidence": float(parts.get("evidence", 0.0)),
        "format": float(parts.get("format", 0.0)),
        "efficiency": float(parts.get("efficiency", 0.0)),
        "strategy": float(parts.get("strategy", 0.0)),
        "reward": float(parts.get("total", 0.0)),
    }
    gold = gold_titles(row.get("supporting_facts"))
    ranked = row.get("retrieved_titles") or []
    if gold and ranked:
        out[f"recall@{rank_k}"] = recall_at_k(ranked, gold, rank_k)
        out[f"mrr@{rank_k}"] = mrr_at_k(ranked, gold, rank_k)
        out[f"ndcg@{rank_k}"] = ndcg_at_k(ranked, gold, rank_k)
    return out


def summarize(records: Iterable[Dict[str, Any]], rank_k: int = RANK_K,
              judge: Any = None, judge_sample: int = 0) -> Dict[str, Any]:
    rows = list(records)
    if not rows:
        return {"count": 0, "overall": {}, "by_source": {}}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        scored = dict(row)
        scored.update(score_row(row, rank_k))
        if judge is not None:
            scored["judge"] = judge.score(row.get("question", ""), last_answer(row.get("text", "")),
                                          _gold_answers(row))
        groups[str(row.get("source", "unknown"))].append(scored)

    def block(items: List[Dict[str, Any]]) -> Dict[str, float]:
        return {
            "count": len(items),
            "em": _mean([x["em"] for x in items]),
            "answer_f1": _mean([x["answer_f1"] for x in items]),
            "judge": _mean([x.get("judge") for x in items]) if any("judge" in x for x in items) else None,
            f"recall@{rank_k}": _mean([x.get(f"recall@{rank_k}") for x in items]),
            f"mrr@{rank_k}": _mean([x.get(f"mrr@{rank_k}") for x in items]),
            f"ndcg@{rank_k}": _mean([x.get(f"ndcg@{rank_k}") for x in items]),
            "evidence": _mean([x["evidence"] for x in items]),
            "format": _mean([x["format"] for x in items]),
            "efficiency": _mean([x["efficiency"] for x in items]),
            "reward": _mean([x["reward"] for x in items]),
            "avg_searches": _mean([float(x.get("search_count", 0)) for x in items]),
            "avg_latency_ms": _mean([float(x.get("latency_ms", 0)) for x in items]),
            "avg_tokens": _mean([float(x.get("tokens", 0)) for x in items]),
        }

    by_source = {source: block(items) for source, items in sorted(groups.items())}
    overall = block([x for items in groups.values() for x in items])
    return {"count": len(rows), "rank_k": rank_k, "overall": overall, "by_source": by_source}
