#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Evaluate one served model variant on the official test sets."""
import argparse
import json
import os
import time

from search_r1_refine.evaluation.judge import Judge
from search_r1_refine.evaluation.online import online_eval

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
p.add_argument("--agent-url", required=True)
p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
p.add_argument("--variant", required=True, help="例如 base / sft / grpo / compare")
p.add_argument("--max-samples", type=int, default=None)
p.add_argument("--rank-k", type=int, default=10)
p.add_argument("--timeout", type=int, default=180)
p.add_argument("--no-replay", action="store_true", help="跳过检索指标重放")
p.add_argument("--judge", action="store_true")
p.add_argument("--judge-model", default=None)
args = p.parse_args()

root = os.path.join(args.data_dir, "processed")
out_dir = os.path.join(args.data_dir, "artifacts", "eval", args.variant)
os.makedirs(out_dir, exist_ok=True)

judge = Judge(model=args.judge_model, enabled=args.judge) if args.judge else None
print(f"[Search-R1] 在线评测开始 variant={args.variant} judge={bool(judge)}", flush=True)
started = time.time()
summary = online_eval(os.path.join(root, "eval.jsonl"), args.agent_url, args.retriever_url,
                      os.path.join(out_dir, "eval_traces.jsonl"),
                      max_samples=args.max_samples, timeout=args.timeout,
                      rank_k=args.rank_k, judge=judge, replay=not args.no_replay)
with open(os.path.join(out_dir, "eval_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
print(f"[Search-R1] 在线评测完成，用时 {time.time() - started:.1f}s", flush=True)
