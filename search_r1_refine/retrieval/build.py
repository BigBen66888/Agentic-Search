"""Index construction entry points.

BM25 is delegated to an external library (``bm25s`` / ``rank_bm25``) and the
dense index is a FAISS IVF-PQ index; nothing is implemented by hand here.
"""
from __future__ import annotations

import json
import os
import time
from typing import Iterator

from .bm25 import BM25Index, available_backends
from .dense import build_dense_index, build_offsets, iter_jsonl, _passage_text


def iter_texts(corpus_path: str) -> Iterator[str]:
    for doc in iter_jsonl(corpus_path):
        yield _passage_text(doc)


def build_bm25(corpus_path: str, output_dir: str, backend: str = "auto",
               k1: float = 1.5, b: float = 0.75) -> dict:
    started = time.time()
    os.makedirs(output_dir, exist_ok=True)
    if backend == "auto" and not available_backends():
        raise ImportError("no BM25 library installed; pip install bm25s")
    texts = list(iter_texts(corpus_path))
    index = BM25Index.build(texts, backend=backend, k1=k1, b=b)
    index.save(output_dir)
    stats = {"backend": index.backend, "documents": len(texts),
             "output_dir": output_dir, "seconds": round(time.time() - started, 1)}
    print(json.dumps({"bm25": stats}, ensure_ascii=False), flush=True)
    return stats


def build_dense(corpus_path: str, index_path: str, offsets_path: str,
                manifest_path: str = None, **kwargs) -> dict:
    return build_dense_index(corpus_path, index_path, offsets_path, manifest_path, **kwargs)


def build_offsets_only(corpus_path: str, offsets_path: str) -> int:
    return build_offsets(corpus_path, offsets_path)
