#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Build the two retrieval ways over the 21M Wikipedia corpus.

    python scripts/build_indices.py --data-dir /data/search_r1 \
        --corpus /data/search_r1/corpus/wiki-18/wiki-18.jsonl --dense --bm25

Dense  -> e5 embeddings + FAISS IVF-PQ (compressed, GPU trained)
Sparse -> BM25 via an external library (bm25s preferred)
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--corpus", default=None, help="默认自动在 data-dir/corpus 下查找 *.jsonl")
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--dense", action="store_true")
    p.add_argument("--bm25", action="store_true")
    p.add_argument("--offsets-only", action="store_true")
    p.add_argument("--nlist", type=int, default=32768)
    p.add_argument("--pq-m", type=int, default=None)
    p.add_argument("--pq-nbits", type=int, default=8)
    p.add_argument("--train-size", type=int, default=1500000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--add-batch", type=int, default=200000)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--bm25-backend", default="auto")
    args = p.parse_args()

    root = args.data_dir
    index_dir = os.path.join(root, "indices")
    os.makedirs(index_dir, exist_ok=True)
    corpus = args.corpus
    if not corpus:
        from pathlib import Path

        matches = sorted(Path(root).joinpath("corpus").rglob("*.jsonl"))
        if not matches:
            raise FileNotFoundError("未找到语料，请先运行 scripts/download_data.py --corpus")
        corpus = str(max(matches, key=lambda x: x.stat().st_size))
    print(f"[Search-R1] 语料：{corpus}", flush=True)

    offsets_path = os.path.join(index_dir, "doc_offsets.npy")
    stats = {"corpus": corpus}

    if args.offsets_only or args.dense or args.bm25:
        if not os.path.exists(offsets_path):
            from search_r1_refine.retrieval.dense import build_offsets

            count = build_offsets(corpus, offsets_path)
            print(f"[Search-R1] 偏移量写入 {offsets_path}（{count} 行）", flush=True)
            stats["offsets"] = count

    if args.bm25:
        from search_r1_refine.retrieval.build import build_bm25

        stats["bm25"] = build_bm25(corpus, os.path.join(index_dir, "bm25"),
                                   backend=args.bm25_backend)

    if args.dense:
        from search_r1_refine.retrieval.dense import build_dense_index

        stats["dense"] = build_dense_index(
            corpus, os.path.join(index_dir, "e5_ivfpq.faiss"), offsets_path,
            manifest_path=os.path.join(index_dir, "doc_manifest.jsonl"),
            model_name=args.model, nlist=args.nlist, pq_m=args.pq_m, pq_nbits=args.pq_nbits,
            train_size=args.train_size, batch_size=args.batch_size, add_batch=args.add_batch,
            use_gpu=not args.cpu, gpu_id=args.gpu_id, max_length=args.max_length)

    with open(os.path.join(index_dir, "index_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
