#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from search_r1_refine.retrieval.build_index import build_dense, read_jsonl, bm25_tokenize

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--corpus", required=True, help="JSONL with doc_id,title,text")
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()
    root = Path(args.data_dir)
    index_dir = root / "indices"
    index_dir.mkdir(parents=True, exist_ok=True)
    print(build_dense(args.corpus, str(index_dir / "e5_flat.faiss"), str(index_dir / "doc_manifest.jsonl"), args.model, args.batch_size))
    try:
        from rank_bm25 import BM25Okapi
        docs = read_jsonl(args.corpus)
        tokenized = [bm25_tokenize(x.get("text", x.get("contents", ""))) for x in docs]
        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(BM25Okapi(tokenized), f)
        print("BM25 index written:", index_dir / "bm25.pkl")
    except ImportError:
        print("rank-bm25 not installed; dense index was still built.")

if __name__ == "__main__":
    main()
