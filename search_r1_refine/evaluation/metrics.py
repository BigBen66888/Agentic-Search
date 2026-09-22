from collections import defaultdict
from typing import Any, Dict, Iterable
from search_r1_refine.rl.reward import score_trajectory

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

def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    if not rows:
        return {"count": 0}
    groups = defaultdict(list)
    for row in rows:
        score = score_trajectory(row.get("text", ""), _gold_answers(row), row.get("supporting_facts"))
        groups[row.get("source", "unknown")].append({**row, **score})
    def avg(items, key):
        return sum(float(x.get(key, 0)) for x in items) / max(len(items), 1)
    by_source = {}
    for source, items in groups.items():
        by_source[source] = {"count": len(items), "answer_f1": avg(items, "answer"), "evidence": avg(items, "evidence"),
                             "format": avg(items, "format"), "efficiency": avg(items, "efficiency"),
                             "reward": avg(items, "total"), "avg_searches": avg(items, "search_count"),
                             "avg_latency_ms": avg(items, "latency_ms"), "avg_tokens": avg(items, "tokens")}
    return {"count": len(rows), "macro_answer_f1": sum(x["answer_f1"] for x in by_source.values()) / max(len(by_source), 1), "by_source": by_source}
