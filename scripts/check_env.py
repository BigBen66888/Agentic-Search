#!/usr/bin/env python3
"""Environment check for the GPU box (run before the long jobs)."""
import importlib.util
import os
import shutil
import sys
from urllib.parse import urlsplit, urlunsplit

EXPECTED_GPU_HINTS = ("4090", "A100", "H100", "A800", "L40", "V100")


def check(name):
    ok = importlib.util.find_spec(name) is not None
    print(f"[Search-R1] {name}: {'OK' if ok else 'MISSING'}", flush=True)
    return ok


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--expect-gpus", type=int, default=4)
    args = p.parse_args()

    print("[Search-R1] 环境检查开始", flush=True)
    print("[Search-R1] Python:", sys.version.replace("\n", " "), flush=True)
    for package in ("torch", "transformers", "faiss", "bm25s", "rank_bm25", "vllm", "ray", "verl", "datasets"):
        check(package)

    if check("torch"):
        import torch

        print(f"[Search-R1] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
        count = torch.cuda.device_count()
        print(f"[Search-R1] gpu_count={count}", flush=True)
        for i in range(count):
            name = torch.cuda.get_device_name(i)
            print(f"[Search-R1] gpu[{i}]={name}", flush=True)
        if count < args.expect_gpus:
            print(f"[Search-R1] 警告：期望 {args.expect_gpus} 张 GPU，实际 {count} 张", flush=True)
        if count and not any(h in torch.cuda.get_device_name(0) for h in EXPECTED_GPU_HINTS):
            print("[Search-R1] 警告：GPU 型号不在已知列表，请确认算力与 dtype 设置", flush=True)

    required = [
        os.path.join(args.data_dir, "processed", "eval.jsonl"),
        os.path.join(args.data_dir, "processed", "train_pool.jsonl"),
        os.path.join(args.data_dir, "indices", "doc_offsets.npy"),
    ]
    for path in required:
        print(f"[Search-R1] {'OK' if os.path.exists(path) else 'MISSING'} {path}", flush=True)
    for name in ("e5_ivfpq.faiss", "bm25", "doc_manifest.jsonl"):
        path = os.path.join(args.data_dir, "indices", name)
        print(f"[Search-R1] {'OK' if os.path.exists(path) else 'MISSING'} {path}", flush=True)

    try:
        import requests

        parts = urlsplit(args.retriever_url)
        url = urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))
        response = requests.get(url, timeout=10)
        print(f"[Search-R1] 检索服务 {url}: {response.json()}", flush=True)
    except Exception as exc:
        print(f"[Search-R1] 检索服务不可达：{exc}", flush=True)

    total, used, free = shutil.disk_usage(args.data_dir)
    print(f"[Search-R1] 磁盘剩余 {free // (1024 ** 3)} GB / 共 {total // (1024 ** 3)} GB", flush=True)
    print("[Search-R1] 环境检查结束", flush=True)


if __name__ == "__main__":
    main()
