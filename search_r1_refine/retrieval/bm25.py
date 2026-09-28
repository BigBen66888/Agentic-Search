"""Sparse retrieval backed by an off-the-shelf BM25 library.

This module deliberately contains **no BM25 implementation**: scoring, idf and
top-k selection are delegated to a third-party library. Two backends are
supported:

``bm25s``       preferred. Written for large corpora, supports fast top-k
                retrieval without materialising a 21M-wide score vector.
``rank_bm25``   pure-python fallback, only sane for small corpora.

Both are persisted with their own save/load helpers so the retrieval service
never re-implements serialisation either.
"""
from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple

BACKEND_BM25S = "bm25s"
BACKEND_RANK = "rank_bm25"


def available_backends() -> List[str]:
    found = []
    try:
        import bm25s  # noqa: F401
        found.append(BACKEND_BM25S)
    except ImportError:
        pass
    try:
        import rank_bm25  # noqa: F401
        found.append(BACKEND_RANK)
    except ImportError:
        pass
    return found


def resolve_backend(backend: str = "auto") -> str:
    """Pick a concrete backend, raising a helpful error when none is installed."""
    found = available_backends()
    if backend in ("auto", ""):
        if not found:
            raise ImportError(
                "No BM25 library installed. Run: pip install bm25s  (or pip install rank-bm25)"
            )
        return found[0]
    if backend not in (BACKEND_BM25S, BACKEND_RANK):
        raise ValueError(f"unknown bm25 backend: {backend}")
    if backend not in found:
        raise ImportError(f"bm25 backend '{backend}' requested but not installed")
    return backend


class BM25Index:
    """Thin wrapper so callers never touch backend-specific APIs."""

    def __init__(self, backend: str, index=None, corpus_tokens=None, size: int = 0):
        self.backend = backend
        self.index = index
        self.corpus_tokens = corpus_tokens
        self.size = size

    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        backend: str = "auto",
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: str = "en",
    ) -> "BM25Index":
        backend = resolve_backend(backend)
        texts = [str(t or "") for t in texts]
        if backend == BACKEND_BM25S:
            import bm25s

            tokens = bm25s.tokenize(texts, stopwords=stopwords, show_progress=False)
            index = bm25s.BM25(k1=k1, b=b)
            index.index(tokens)
            return cls(backend, index=index, size=len(texts))
        from rank_bm25 import BM25Okapi

        tokens = [t.lower().split() for t in texts]
        return cls(backend, index=BM25Okapi(tokens, k1=k1, b=b), corpus_tokens=tokens, size=len(texts))

    def save(self, directory: str) -> str:
        os.makedirs(directory, exist_ok=True)
        if self.backend == BACKEND_BM25S:
            # corpus text lives in the original JSONL; keep the index small
            self.index.save(directory, allow_pickle=True)
        else:
            import pickle

            with open(os.path.join(directory, "rank_bm25.pkl"), "wb") as f:
                pickle.dump({"index": self.index, "tokens": self.corpus_tokens}, f)
        with open(os.path.join(directory, "bm25_meta.json"), "w", encoding="utf-8") as f:
            json.dump({"backend": self.backend, "size": self.size}, f, ensure_ascii=False, indent=2)
        return directory

    @classmethod
    def load(cls, directory: str, backend: str = "auto", mmap: bool = True) -> "BM25Index":
        with open(os.path.join(directory, "bm25_meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        backend = meta.get("backend") if backend in ("auto", "") else backend
        if backend == BACKEND_BM25S:
            import bm25s

            index = bm25s.BM25.load(directory, load_corpus=False, mmap=mmap)
            return cls(backend, index=index, size=int(meta.get("size", 0)))
        import pickle

        with open(os.path.join(directory, "rank_bm25.pkl"), "rb") as f:
            payload = pickle.load(f)
        return cls(backend, index=payload["index"], corpus_tokens=payload["tokens"],
                   size=int(meta.get("size", 0)))

    def retrieve(self, queries: Sequence[str], k: int = 10) -> List[List[Tuple[int, float]]]:
        """Return ``[(doc_id, score), ...]`` per query, best first."""
        if not queries:
            return []
        if self.backend == BACKEND_BM25S:
            import bm25s

            q_tokens = bm25s.tokenize([str(q) for q in queries], show_progress=False)
            doc_ids, scores = self.index.retrieve(q_tokens, k=min(k, max(self.size, 1)))
            out = []
            for row_ids, row_scores in zip(doc_ids, scores):
                out.append([(int(i), float(s)) for i, s in zip(row_ids, row_scores) if int(i) >= 0])
            return out
        results = []
        for query in queries:
            raw = self.index.get_scores(str(query).lower().split())
            order = sorted(range(len(raw)), key=lambda i: raw[i], reverse=True)[:k]
            results.append([(int(i), float(raw[i])) for i in order])
        return results
