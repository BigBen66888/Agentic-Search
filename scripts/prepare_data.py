#!/usr/bin/env python3
import argparse
import os
import time
from search_r1_refine.data.prepare import prepare, SOURCES

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
p.add_argument("--dataset-name", default="RUC-NLPIR/FlashRAG_datasets")
p.add_argument("--sources", nargs="+", default=SOURCES)
p.add_argument("--eval-split", default="test")
p.add_argument("--rewrite-model", default=None)
p.add_argument("--max-variants", type=int, default=2)
p.add_argument("--rewrite-batch-size", type=int, default=8)
p.add_argument("--max-total-multiplier", type=float, default=2.5)
p.add_argument("--easy-max", type=int, default=1)
p.add_argument("--medium-max", type=int, default=3)
args = p.parse_args()
args.output_dir = os.path.join(args.data_dir, "processed")
args.data_dir = args.data_dir
print(f"[Search-R1] 数据处理开始：sources={args.sources}", flush=True)
print("[Search-R1] 阶段：质量过滤 -> 查询改写（可选） -> 难度分层 -> JSONL/Parquet", flush=True)
started = time.time()
print(prepare(args))
print(f"[Search-R1] 数据处理完成，用时 {time.time() - started:.1f}s", flush=True)
