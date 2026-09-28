#!/usr/bin/env python3
"""Prepare NQ + HotpotQA evaluation sets, train pool and optional augmentation."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import json
import os
import time

from search_r1_refine.data.prepare import build_arg_parser, parse_eval_splits, prepare

p = build_arg_parser()
args = p.parse_args()
if not args.data_dir:
    p.error("--data-dir 必填")
args.output_dir = os.path.join(args.data_dir, "processed")
args.eval_splits = parse_eval_splits(args.eval_splits)

print("[Search-R1] 数据处理开始：nq/hotpotqa，质量过滤 -> 难度分层 -> 查询扩展", flush=True)
print(f"[Search-R1] 评测 split：{ {s: args.eval_splits.get(s, args.eval_split) for s in args.sources} }", flush=True)
started = time.time()
stats = prepare(args)
print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
print(f"[Search-R1] 数据处理完成，用时 {time.time() - started:.1f}s", flush=True)
if not stats["eval_sources"]:
    raise SystemExit("没有任何数据集通过校验，请检查下载的数据文件与 data_stats.json")
