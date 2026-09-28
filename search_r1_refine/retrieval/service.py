"""Two-way RRF retrieval service.

Runs on its own GPU box: dense (e5 + FAISS IVF-PQ) and sparse (BM25) are two
independent retrieval ways, fused with Reciprocal Rank Fusion. The training box
talks to it over HTTP, so the 21M index never competes with RL for GPU memory.
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from .rrf import reciprocal_rank_fusion

app = FastAPI(title="Search-R1 Refine RRF Retriever")
STATE = {}


class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_text: bool = True
    max_chars: int = 800


def _as_dicts(hits):
    return [{"doc_id": str(i), "score": float(s)} for i, s in hits]


class HybridRetriever:
    def __init__(self, dense, bm25, store, rrf_k=60.0, dense_topk=100, bm25_topk=100):
        self.dense = dense
        self.bm25 = bm25
        self.store = store
        self.rrf_k = rrf_k
        self.dense_topk = dense_topk
        self.bm25_topk = bm25_topk
        if dense is None and bm25 is None:
            raise RuntimeError("at least one retrieval way must be available")

    @property
    def ways(self) -> List[str]:
        names = []
        if self.dense is not None:
            names.append("dense")
        if self.bm25 is not None:
            names.append("bm25")
        return names

    def search(self, query: str, topk: int = 10, return_text: bool = True,
               max_chars: int = 800, ways: Optional[str] = None):
        ways = ways or ("both" if len(self.ways) == 2 else self.ways[0])
        if ways not in ("both", "dense", "bm25"):
            raise ValueError(f"unknown retrieval way: {ways}")
        needed = {"both": {"dense", "bm25"}, "dense": {"dense"}, "bm25": {"bm25"}}[ways]
        if not needed.issubset(self.ways):
            raise ValueError(f"requested {ways}, available {self.ways}")
        lists = []
        if ways in ("both", "bm25"):
            lists.append(_as_dicts(self.bm25.retrieve([query], k=self.bm25_topk)[0]))
        if ways in ("both", "dense"):
            lists.append(_as_dicts(self.dense.search([query], k=self.dense_topk)[0]))
        fused = reciprocal_rank_fusion(lists, k=self.rrf_k, topk=topk)
        if return_text and fused:
            docs = {d["row"]: d for d in self.store.read([int(f["doc_id"]) for f in fused], max_chars=max_chars)}
            for item in fused:
                doc = docs.get(int(item["doc_id"]), {})
                item["title"] = doc.get("title", "")
                item["text"] = doc.get("text", "")
        return fused

    def batch_search(self, queries, topk=10, return_text=True, max_chars=800,
                     ways=None):
        return [self.search(q, topk, return_text, max_chars, ways) for q in queries]


@app.get("/health")
def health():
    retriever = STATE.get("retriever")
    return {
        "status": "ok" if retriever else "not_ready",
        "fusion": "rrf",
        "rrf_k": STATE.get("rrf_k"),
        "ways": retriever.ways if retriever else [],
        "passages": len(STATE["store"]) if STATE.get("store") is not None else 0,
        "gpu": STATE.get("gpu_id"),
    }


@app.post("/retrieve")
def retrieve(req: RetrieveRequest, ways: Optional[str] = Query(default=None, pattern="^(both|dense|bm25)$")):
    started = time.perf_counter()
    retriever = STATE["retriever"]
    ways = ways or ("both" if len(retriever.ways) == 2 else retriever.ways[0])
    topk = req.topk or 10
    try:
        results = retriever.batch_search(req.queries, topk=topk,
                                         return_text=req.return_text, max_chars=req.max_chars,
                                         ways=ways)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": results, "ways": ways,
            "latency_ms": (time.perf_counter() - started) * 1000}


def build_retriever(args) -> HybridRetriever:
    from .bm25 import BM25Index
    from .dense import DenseIndex, PassageStore

    offsets = np.load(args.offsets, mmap_mode="r")
    store = PassageStore(args.corpus, offsets)
    ways = getattr(args, "ways", "both")
    dense = None
    if ways in ("both", "dense") and args.dense_index and os.path.isfile(args.dense_index):
        dense = DenseIndex.load(args.dense_index, args.offsets, args.corpus,
                                model_name=args.model, nprobe=args.nprobe,
                                gpu_id=args.gpu_id, use_gpu=not args.cpu,
                                max_length=args.max_length)
    bm25 = None
    if ways in ("both", "bm25") and args.bm25_index and os.path.isfile(
        os.path.join(args.bm25_index, "bm25_meta.json")
    ):
        bm25 = BM25Index.load(args.bm25_index, backend=args.bm25_backend)
    required = {"both": {"dense", "bm25"}, "dense": {"dense"}, "bm25": {"bm25"}}[ways]
    loaded = {name for name, index in (("dense", dense), ("bm25", bm25)) if index is not None}
    if not required.issubset(loaded):
        raise FileNotFoundError(
            f"requested retrieval ways {sorted(required)}, loaded {sorted(loaded)}; "
            "build the missing index before starting the service"
        )
    return HybridRetriever(dense, bm25, store, rrf_k=args.rrf_k,
                           dense_topk=args.dense_topk, bm25_topk=args.bm25_topk)


def main():
    p = argparse.ArgumentParser(description="Two-way RRF retrieval service (dense + BM25)")
    p.add_argument("--corpus", required=True)
    p.add_argument("--offsets", required=True)
    p.add_argument("--dense-index", default=None)
    p.add_argument("--bm25-index", default=None)
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--bm25-backend", default="auto")
    p.add_argument("--ways", choices=["both", "dense", "bm25"], default="both",
                   help="启用哪几路召回：both=双路 RRF，dense/bm25=单路")
    p.add_argument("--nprobe", type=int, default=64)
    p.add_argument("--rrf-k", type=float, default=60.0)
    p.add_argument("--dense-topk", type=int, default=100)
    p.add_argument("--bm25-topk", type=int, default=100)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--cpu", action="store_true", help="force CPU dense search")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()

    STATE["rrf_k"] = args.rrf_k
    STATE["retriever"] = build_retriever(args)
    STATE["store"] = STATE["retriever"].store
    dense = STATE["retriever"].dense
    STATE["gpu_id"] = args.gpu_id if dense is not None and dense.use_gpu else None
    print(f"[Search-R1] 检索服务就绪 ways={STATE['retriever'].ways} port={args.port}", flush=True)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
