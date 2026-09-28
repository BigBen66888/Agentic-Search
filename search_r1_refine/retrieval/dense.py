"""Dense retrieval over a 21M-passage Wikipedia corpus.

The index is an IVF-PQ compressed FAISS index (inner product on L2-normalised
e5 embeddings). Float vectors for 21M x 768 would cost ~64 GB, so only a
sample is ever materialised in memory: the corpus is streamed, encoded batch by
batch, and added to the index incrementally.

Passage text is *not* duplicated. We store byte offsets into the original
corpus JSONL so the retrieval service can seek and read only the hits.
"""
from __future__ import annotations

import json
import os
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

VALID_PQ_M = (8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128)


def pick_pq_m(dim: int, preferred: Optional[int] = None) -> int:
    if preferred:
        if preferred not in VALID_PQ_M:
            raise ValueError(f"pq_m must be one of {VALID_PQ_M}, got {preferred}")
        return preferred
    target = dim // 8
    valid = [m for m in VALID_PQ_M if m <= target and dim % m == 0]
    return max(valid) if valid else min(VALID_PQ_M)


def count_lines(path: str) -> int:
    total = 0
    with open(path, "rb") as f:
        for _ in f:
            total += 1
    return total


def build_offsets(corpus_path: str, output_path: str, flush_every: int = 500000) -> int:
    """Record the byte offset of every JSONL line so hits can be read lazily."""
    import numpy as np

    offsets: List[int] = []
    chunks: List["np.ndarray"] = []
    with open(corpus_path, "rb") as f:
        position = 0
        while True:
            line = f.readline()
            if not line:
                break
            offsets.append(position)
            position += len(line)
            if len(offsets) >= flush_every:
                chunks.append(np.asarray(offsets, dtype=np.int64))
                offsets = []
    if offsets:
        chunks.append(np.asarray(offsets, dtype=np.int64))
    array = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, array)
    return int(array.shape[0])


def iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _passage_text(doc: dict) -> str:
    return str(doc.get("text") or doc.get("contents") or doc.get("passage") or "")


class Encoder:
    """e5 encoder. Query side uses the ``query: `` prefix, passage side ``passage: ``."""

    def __init__(self, model_name: str, device: str = "auto", max_length: int = 256, fp16: bool = True):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if device.startswith("cuda") and fp16:
            self.model = self.model.half()
        self.model.to(device).eval()

    @staticmethod
    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(self, texts: Sequence[str], prefix: str, batch_size: int = 256):
        import numpy as np

        out = []
        for start in range(0, len(texts), batch_size):
            batch = [f"{prefix}{t}" for t in texts[start:start + batch_size]]
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True,
                                    truncation=True, max_length=self.max_length).to(self.device)
            with self.torch.no_grad():
                hidden = self.model(**tokens).last_hidden_state
            emb = self.torch.nn.functional.normalize(self.mean_pool(hidden, tokens["attention_mask"]), dim=-1)
            out.append(emb.float().cpu().numpy().astype("float32"))
        return np.concatenate(out, axis=0) if out else np.zeros((0, 0), dtype="float32")


def sample_training_vectors(corpus_path: str, encoder: Encoder, sample_size: int,
                            total: int, batch_size: int = 256, seed: int = 0):
    """Collect a float32 sample used to train the IVF quantiser."""
    import numpy as np

    rng = np.random.default_rng(seed)
    keep = set(rng.choice(total, size=min(sample_size, total), replace=False).tolist()) if total else set()
    chunks, buffer = [], []
    for index, doc in enumerate(iter_jsonl(corpus_path)):
        if index in keep:
            buffer.append(_passage_text(doc))
        if len(buffer) >= batch_size:
            chunks.append(encoder.encode(buffer, "passage: ", batch_size))
            buffer = []
    if buffer:
        chunks.append(encoder.encode(buffer, "passage: ", batch_size))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 768), dtype="float32")


def train_ivf_pq(sample, dim: int, nlist: int, pq_m: int, pq_nbits: int,
                 use_gpu: bool, gpu_id: int = 0):
    import faiss

    factory = f"IVF{nlist},PQ{pq_m}x{pq_nbits}"
    index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
    if use_gpu:
        try:
            resources = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(resources, gpu_id, index)
            gpu_index.train(sample)
            return faiss.index_gpu_to_cpu(gpu_index)
        except Exception as exc:  # fall back to CPU training rather than crash a long job
            print(f"[Search-R1] GPU IVF-PQ training unavailable ({exc}); falling back to CPU", flush=True)
    index.train(sample)
    return index


def build_dense_index(corpus_path: str, index_path: str, offsets_path: str,
                      manifest_path: Optional[str] = None,
                      model_name: str = "intfloat/e5-base-v2",
                      nlist: int = 32768, pq_m: Optional[int] = None, pq_nbits: int = 8,
                      train_size: int = 1500000, batch_size: int = 256,
                      add_batch: int = 200000, use_gpu: bool = True, gpu_id: int = 0,
                      max_length: int = 256, sample_every: int = 1) -> dict:
    """Encode the corpus stream and build a compressed IVF-PQ index."""
    import faiss
    import numpy as np

    print("[Search-R1] 计数语料行数", flush=True)
    total = count_lines(corpus_path)
    print(f"[Search-R1] 语料行数={total}", flush=True)
    encoder = Encoder(model_name, max_length=max_length)
    dim = int(encoder.model.config.hidden_size)
    pq_m = pick_pq_m(dim, pq_m)
    print(f"[Search-R1] dim={dim} nlist={nlist} pq_m={pq_m} nbits={pq_nbits}", flush=True)

    actual_train = min(train_size, max(total // sample_every, 1))
    print(f"[Search-R1] 采样训练向量 {actual_train}", flush=True)
    sample = sample_training_vectors(corpus_path, encoder, actual_train, total, batch_size)
    print(f"[Search-R1] 训练 IVF-PQ，sample={sample.shape}", flush=True)
    index = train_ivf_pq(sample, dim, nlist, pq_m, pq_nbits, use_gpu, gpu_id)
    del sample

    print("[Search-R1] 流式编码并写入索引", flush=True)
    buffer, pending, written = [], [], 0
    for doc in iter_jsonl(corpus_path):
        buffer.append(_passage_text(doc))
        if len(buffer) >= batch_size:
            pending.append(encoder.encode(buffer, "passage: ", batch_size))
            buffer = []
        if len(pending) * batch_size >= add_batch:
            index.add(np.concatenate(pending, axis=0))
            written += sum(x.shape[0] for x in pending)
            pending = []
            print(f"[Search-R1] 已写入 {written}/{total}", flush=True)
    if buffer:
        pending.append(encoder.encode(buffer, "passage: ", batch_size))
    if pending:
        index.add(np.concatenate(pending, axis=0))
        written += sum(x.shape[0] for x in pending)

    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
    faiss.write_index(index, index_path)
    build_offsets(corpus_path, offsets_path)
    stats = {"documents": int(written), "dimension": dim, "nlist": nlist,
             "pq_m": pq_m, "pq_nbits": pq_nbits, "train_size": int(actual_train),
             "index_path": index_path, "is_trained": bool(index.is_trained)}
    if manifest_path:
        with open(manifest_path, "w", encoding="utf-8") as out:
            for i, doc in enumerate(iter_jsonl(corpus_path)):
                out.write(json.dumps({"doc_id": str(doc.get("doc_id", i)),
                                      "title": str(doc.get("title", ""))}, ensure_ascii=False) + "\n")
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    return stats


class PassageStore:
    """Random-access reader over the corpus JSONL using precomputed byte offsets."""

    def __init__(self, corpus_path: str, offsets):
        self.corpus_path = corpus_path
        self.offsets = offsets

    def __len__(self) -> int:
        return int(len(self.offsets))

    def read(self, doc_ids: Iterable[int], max_chars: int = 0) -> List[dict]:
        result = []
        # Offsets are byte positions recorded from the original JSONL. Read in
        # binary mode so seek() has the same meaning on every platform.
        with open(self.corpus_path, "rb") as f:
            for i in doc_ids:
                i = int(i)
                if i < 0 or i >= len(self.offsets):
                    continue
                f.seek(int(self.offsets[i]))
                try:
                    doc = json.loads(f.readline().decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                text = _passage_text(doc)
                if max_chars and len(text) > max_chars:
                    text = text[:max_chars]
                result.append({"doc_id": str(doc.get("doc_id", i)), "row": i,
                               "title": str(doc.get("title", "")), "text": text})
        return result


class DenseIndex:
    """Query-side wrapper: e5 encoder + IVF-PQ index, optionally resident on GPU."""

    def __init__(self, index, encoder: Encoder, store: PassageStore,
                 nprobe: int = 64, gpu_id: int = 0, use_gpu: bool = True,
                 gpu_resources=None):
        self.index = index
        self.encoder = encoder
        self.store = store
        self.nprobe = nprobe
        self.gpu_id = gpu_id
        self.nlist = int(getattr(index, "nlist", 0))
        self.use_gpu = use_gpu
        self.gpu_resources = gpu_resources
        try:
            index.nprobe = min(nprobe, max(self.nlist, 1))
        except Exception:
            pass

    @classmethod
    def load(cls, index_path: str, offsets_path: str, corpus_path: str,
             model_name: str = "intfloat/e5-base-v2", nprobe: int = 64,
             gpu_id: int = 0, use_gpu: bool = True, max_length: int = 256) -> "DenseIndex":
        import faiss
        import numpy as np

        index = faiss.read_index(index_path)
        resources = None
        if use_gpu:
            try:
                resources = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(resources, gpu_id, index)
            except Exception as exc:
                print(f"[Search-R1] GPU 索引不可用（{exc}），使用 CPU 检索", flush=True)
                use_gpu = False
                resources = None
        try:
            index.nprobe = min(nprobe, max(int(getattr(index, "nlist", 1)), 1))
        except Exception:
            pass
        offsets = np.load(offsets_path, mmap_mode="r")
        encoder = Encoder(model_name, device=f"cuda:{gpu_id}" if use_gpu else "cpu", max_length=max_length)
        store = PassageStore(corpus_path, offsets)
        return cls(index, encoder, store, nprobe, gpu_id, use_gpu, resources)

    def search(self, queries: Sequence[str], k: int = 10) -> List[List[Tuple[int, float]]]:
        import numpy as np

        if not queries:
            return []
        emb = self.encoder.encode(list(queries), "query: ", batch_size=min(64, len(queries)))
        scores, ids = self.index.search(emb, k)
        out = []
        for row_scores, row_ids in zip(scores, ids):
            out.append([(int(i), float(s)) for i, s in zip(row_ids, row_scores) if int(i) >= 0])
        return out

    def read_docs(self, doc_ids: Iterable[int], max_chars: int = 0) -> List[dict]:
        return self.store.read(doc_ids, max_chars=max_chars)
