#!/usr/bin/env python3
"""Build / extend the success-trajectory FIFO buffer.

Sources, in priority order:
  * ``--from-sft``    SFT cold-start trajectories (all of them scored high)
  * ``--from-traces`` online evaluation traces, keeping rows above ``--min-reward``

    python scripts/update_fifo.py --data-dir /data/search_r1 --from-sft
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os

from search_r1_refine.data.schema import read_jsonl
from search_r1_refine.rl.fifo import SuccessFIFO

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
p.add_argument("--from-sft", action="store_true")
p.add_argument("--from-traces", nargs="*", default=None,
               help="eval trace 文件路径，例如 artifacts/eval/sft/eval_traces.jsonl")
p.add_argument("--min-reward", type=float, default=0.8)
p.add_argument("--fifo-path", default=None)
p.add_argument("--max-size", type=int, default=20000)
args = p.parse_args()

fifo_path = args.fifo_path or os.path.join(args.data_dir, "processed", "fifo.jsonl")
buffer = SuccessFIFO.load(fifo_path, max_size=args.max_size)
before = len(buffer)

added = 0
if args.from_sft:
    path = os.path.join(args.data_dir, "processed", "sft_train.jsonl")
    if os.path.exists(path):
        for row in read_jsonl(path):
            record = {
                "sample_id": row.get("sample_id"),
                "source": row.get("source"),
                "question": row.get("question"),
                "answers": row.get("answers", []),
                "prompt_text": row.get("prompt_text") or row.get("question"),
                "messages": row.get("messages", []),
                "reward_total": float(row.get("reward_total", 1.0)),
                "success": True,
            }
            added += 1 if buffer.add(record) else 0
    else:
        print(f"[Search-R1] 未找到 {path}", flush=True)

for trace_path in args.from_traces or []:
    if not os.path.exists(trace_path):
        print(f"[Search-R1] 未找到 {trace_path}", flush=True)
        continue
    from search_r1_refine.evaluation.metrics import score_row

    for row in read_jsonl(trace_path):
        scored = score_row(row)
        if float(scored.get("reward", 0.0)) < args.min_reward:
            continue
        record = {
            "sample_id": row.get("sample_id"),
            "source": row.get("source"),
            "question": row.get("question"),
            "answers": row.get("answers", []),
            "prompt_text": row.get("prompt_text") or row.get("question"),
            "text": row.get("text", ""),
            "reward_total": float(scored["reward"]),
            "success": True,
        }
        added += 1 if buffer.add(record) else 0

buffer.save(fifo_path)
print(json.dumps({"fifo_path": fifo_path, "before": before, "added": added, "after": len(buffer)},
                 ensure_ascii=False), flush=True)
