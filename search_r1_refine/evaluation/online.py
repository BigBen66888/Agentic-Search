"""Online evaluation against a served agent.

The agent is a black box exposing ``POST /generate``. Retrieval metrics are
computed by replaying the agent's own ``<search>`` queries against the *same*
RRF service used in training, so retrieval quality is comparable across runs.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import requests

from search_r1_refine.data.schema import read_jsonl, write_jsonl
from search_r1_refine.rl.reward import blocks

from .metrics import _prompt_of, summarize


def call_agent(url: str, prompt: str, timeout: int = 120) -> Dict:
    response = requests.post(url, json={"prompt": prompt}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {"text": str(payload)}


def replay_retrieval(retriever_url: str, searches: List[str], topk: int = 10,
                     timeout: int = 60, max_chars: int = 0) -> List[str]:
    """Re-run the agent's queries through the RRF service and return titles."""
    if not searches or not retriever_url:
        return []
    titles: List[str] = []
    try:
        response = requests.post(retriever_url,
                                 json={"queries": searches, "topk": topk,
                                       "return_text": False, "max_chars": max_chars},
                                 timeout=timeout)
        response.raise_for_status()
        for ranking in (response.json() or {}).get("result", []):
            titles.extend(str(doc.get("title", "")) for doc in ranking)
    except Exception as exc:
        print(f"[Search-R1] 检索重放失败：{exc}", flush=True)
    return titles


def online_eval(eval_path: str, agent_url: str, retriever_url: str, output_path: str,
                max_samples: int = None, timeout: int = 120, rank_k: int = 10,
                judge=None, sources: List[str] = None, replay: bool = True) -> Dict:
    rows = read_jsonl(eval_path)
    if sources:
        rows = [r for r in rows if r.get("source") in sources]
    if max_samples:
        rows = rows[:max_samples]
    traces = []
    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        try:
            payload = call_agent(agent_url, _prompt_of(row), timeout)
        except Exception as exc:
            print(f"[Search-R1] 第 {index} 条推理失败：{exc}", flush=True)
            continue
        text = str(payload.get("text", "") or "")
        searches = blocks(text, "search")
        trace = dict(row)
        trace.update({
            "text": text,
            "search_count": len(searches),
            "searches": searches,
            "retrieved_titles": payload.get("retrieved_titles")
            or (replay_retrieval(retriever_url, searches, topk=rank_k) if replay else []),
            "latency_ms": (time.perf_counter() - started) * 1000,
            "tokens": len(text.split()),
        })
        traces.append(trace)
        if index % 20 == 0:
            print(f"[Search-R1] 评测进度 {index}/{len(rows)}", flush=True)
    write_jsonl(traces, output_path)
    return summarize(traces, rank_k=rank_k, judge=judge)
