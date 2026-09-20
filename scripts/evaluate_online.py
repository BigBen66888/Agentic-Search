#!/usr/bin/env python3
import argparse
import json
import os
import time
from search_r1_refine.evaluation.online import online_eval

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
p.add_argument("--agent-url", required=True)
p.add_argument("--retriever-url", required=True)
p.add_argument("--max-samples", type=int, default=None)
args = p.parse_args()
print("[Search-R1] 在线评测开始：固定 eval/test 数据，不回传梯度", flush=True)
started = time.time()
root = os.path.join(args.data_dir, "processed")
artifact_dir = os.path.join(args.data_dir, "artifacts")
os.makedirs(artifact_dir, exist_ok=True)
trace_path = os.path.join(artifact_dir, "eval_traces.jsonl")
summary = online_eval(os.path.join(root, "eval.jsonl"), args.agent_url, args.retriever_url, trace_path, args.max_samples)
with open(os.path.join(artifact_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print(f"[Search-R1] 在线评测完成，用时 {time.time() - started:.1f}s", flush=True)
