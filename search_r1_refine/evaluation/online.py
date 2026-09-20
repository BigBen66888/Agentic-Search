import json
import time
import requests
from .metrics import summarize
from search_r1_refine.data.schema import read_jsonl

def call_agent(url, prompt, timeout):
    response = requests.post(url, json={"prompt": prompt}, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("text", payload.get("response", "")))

def online_eval(eval_path, agent_url, retriever_url, output_path, max_samples=None, timeout=60):
    rows = read_jsonl(eval_path)
    if max_samples:
        rows = rows[:max_samples]
    traces = []
    for row in rows:
        started = time.perf_counter()
        prompt = row.get("prompt_text") or row.get("prompt")
        if isinstance(prompt, list):
            prompt = prompt[-1].get("content", "") if prompt else ""
        text = call_agent(agent_url, prompt, timeout)
        traces.append({**row, "text": text, "search_count": text.lower().count("<search>"),
                       "latency_ms": (time.perf_counter() - started) * 1000,
                       "tokens": len(text.split()), "retriever_url": retriever_url})
    with open(output_path, "w", encoding="utf-8") as f:
        for row in traces:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summarize(traces)
