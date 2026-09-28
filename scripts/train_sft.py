#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""SFT cold start on the sampled teacher trajectories.

Single node multi-GPU is handled by torchrun:

    torchrun --nproc_per_node=4 scripts/train_sft.py --data-dir /data/search_r1 \
        --model Qwen/Qwen2.5-3B

Without torchrun the script falls back to single-process training.
"""
import os
import sys

from search_r1_refine.sft.train import build_arg_parser, train

p = build_arg_parser()
p.add_argument("--data-dir", required=True)
args = p.parse_args()
if not args.data:
    args.data = os.path.join(args.data_dir, "processed", "sft_train.jsonl")
if not args.output_dir:
    args.output_dir = os.path.join(args.data_dir, "models", "sft")

if not os.path.exists(args.data):
    raise SystemExit(f"缺少 SFT 数据 {args.data}，请先运行 scripts/generate_sft_data.py")

world = int(os.environ.get("WORLD_SIZE", "1"))
print(f"[Search-R1] SFT 训练开始：model={args.model} data={args.data} world_size={world}", flush=True)
train(args)
print(f"[Search-R1] SFT 完成，模型保存在 {args.output_dir}", flush=True)
