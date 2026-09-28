#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Entry point for the standalone retrieval box (its own GPU)."""
import argparse

from search_r1_refine.retrieval.service import main as service_main


def main():
    p = argparse.ArgumentParser(description="启动二路 RRF 检索服务（dense IVF-PQ + BM25）")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--corpus", default=None)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--rrf-k", type=float, default=60.0)
    p.add_argument("--dense-topk", type=int, default=100)
    p.add_argument("--bm25-topk", type=int, default=100)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--bm25-backend", default="auto")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--ways", choices=["both", "dense", "bm25"], default="both",
                   help="both=双路 RRF；dense/bm25=单路，用于单路/多路消融")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    import os
    from pathlib import Path

    index_dir = os.path.join(args.data_dir, "indices")
    if not os.path.isfile(os.path.join(index_dir, "doc_offsets.npy")):
        raise FileNotFoundError("未找到 indices/doc_offsets.npy；请先运行 scripts/build_indices.py")
    corpus = args.corpus
    if not corpus:
        matches = sorted(Path(args.data_dir).joinpath("corpus").rglob("*.jsonl"))
        if not matches:
            raise FileNotFoundError("未找到语料")
        corpus = str(max(matches, key=lambda x: x.stat().st_size))

    import sys

    sys.argv = [
        "serve_retriever",
        "--corpus", corpus,
        "--offsets", os.path.join(index_dir, "doc_offsets.npy"),
        "--dense-index", os.path.join(index_dir, "e5_ivfpq.faiss"),
        "--bm25-index", os.path.join(index_dir, "bm25"),
        "--model", args.model,
        "--nprobe", str(args.nprobe),
        "--rrf-k", str(args.rrf_k),
        "--dense-topk", str(args.dense_topk),
        "--bm25-topk", str(args.bm25_topk),
        "--gpu-id", str(args.gpu_id),
        "--bm25-backend", args.bm25_backend,
        "--max-length", str(args.max_length),
        "--ways", args.ways,
        "--port", str(args.port),
        "--host", args.host,
    ]
    if args.cpu:
        sys.argv.append("--cpu")
    service_main()


if __name__ == "__main__":
    main()
