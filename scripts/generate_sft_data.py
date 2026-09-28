#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Sample high-scoring teacher trajectories for SFT cold start (default 2k)."""
import os
import sys

from search_r1_refine.sft.generate import build_arg_parser, generate

p = build_arg_parser()
p.add_argument("--data-dir", required=True)
args, unknown = p.parse_known_args()
if not args.train_pool:
    args.train_pool = os.path.join(args.data_dir, "processed", "train_pool.jsonl")
if not args.output:
    args.output = os.path.join(args.data_dir, "processed", "sft_train.jsonl")
if not args.stats:
    args.stats = os.path.join(args.data_dir, "processed", "sft_stats.json")

if not os.path.exists(args.train_pool):
    raise SystemExit(f"缺少训练池 {args.train_pool}，请先运行 scripts/prepare_data.py")
if not (args.api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")):
    print("[Search-R1] 警告：未设置 LLM_API_KEY，教师 API 调用可能失败", flush=True)

print(f"[Search-R1] SFT 采样开始：teacher={args.teacher_model} target={args.target}", flush=True)
generate(args)
