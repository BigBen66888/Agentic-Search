import argparse
import json
import os
import pickle
import time
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from .rrf import reciprocal_rank_fusion

class RetrieveRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

class HybridRetriever:
    def __init__(self, dense_index_path, manifest_path, bm25_index_path=None, model_name="intfloat/e5-base-v2", rrf_k=60.0, bm25_topk=20, dense_topk=20):
        self.rrf_k, self.bm25_topk, self.dense_topk = rrf_k, bm25_topk, dense_topk
        self.docs = self._read_jsonl(manifest_path)
        self.bm25 = None
        self.dense = None
        if bm25_index_path and os.path.exists(bm25_index_path):
            with open(bm25_index_path, "rb") as f:
                self.bm25 = pickle.load(f)
        if dense_index_path and os.path.exists(dense_index_path):
            import faiss
            from transformers import AutoModel, AutoTokenizer
            import torch
            self.dense = faiss.read_index(dense_index_path)
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device).eval()
        if self.bm25 is None and self.dense is None:
            raise FileNotFoundError("No BM25 or dense index found")

    @staticmethod
    def _read_jsonl(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]

    def _dense_search(self, query, topk):
        from .build_index import mean_pool
        tokens = self.tokenizer([f"query: {query}"], return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with self.torch.no_grad():
            out = self.model(**tokens)
            emb = self.torch.nn.functional.normalize(mean_pool(out.last_hidden_state, tokens["attention_mask"]), dim=-1).cpu().numpy()
        scores, ids = self.dense.search(emb, topk)
        return [dict(self.docs[int(i)], dense_score=float(s)) for s, i in zip(scores[0], ids[0]) if int(i) >= 0]

    def _bm25_search(self, query, topk):
        scores = self.bm25.get_scores(query.lower().split())
        ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]
        return [dict(self.docs[i], bm25_score=float(scores[i])) for i in ids]

    def search(self, query, topk=10):
        lists = []
        if self.bm25 is not None:
            lists.append(self._bm25_search(query, self.bm25_topk))
        if self.dense is not None:
            lists.append(self._dense_search(query, self.dense_topk))
        return reciprocal_rank_fusion(lists, k=self.rrf_k, topk=topk)

    def batch_search(self, queries, topk=10):
        return [self.search(q, topk) for q in queries]

app = FastAPI(title="Search-R1 Refine RRF Retriever")
retriever = None

@app.get("/health")
def health():
    return {"status": "ok" if retriever else "not_ready", "fusion": "rrf", "rrf_k": retriever.rrf_k if retriever else None}

@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    started = time.perf_counter()
    results = retriever.batch_search(req.queries, req.topk or 10)
    return {"result": results, "latency_ms": (time.perf_counter() - started) * 1000}

def main():
    global retriever
    p = argparse.ArgumentParser()
    p.add_argument("--dense-index", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--bm25-index")
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--rrf-k", type=float, default=60.0)
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    retriever = HybridRetriever(args.dense_index, args.manifest, args.bm25_index, args.model, args.rrf_k)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()

