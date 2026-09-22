import argparse
import json
import os

def bm25_tokenize(text):
    """Stable whitespace + CJK unigram/bigram tokenizer for BM25."""
    import re
    tokens = []
    for part in re.findall(r"\S+", str(text).lower()):
        if re.search(r"[\u3400-\u9fff]", part):
            chars = re.findall(r"[\u3400-\u9fff]", part)
            tokens.extend(chars)
            tokens.extend(a + b for a, b in zip(chars, chars[1:]))
            tokens.extend(x for x in re.split(r"[^\w]+", part) if x and not re.search(r"[\u3400-\u9fff]", x))
        else:
            tokens.append(part)
    return tokens

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]

def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

def build_dense(corpus_path, output_index, manifest_path, model_name="intfloat/e5-base-v2", batch_size=64, max_length=256):
    import numpy as np
    import torch
    import faiss
    from transformers import AutoModel, AutoTokenizer
    docs = read_jsonl(corpus_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    vectors = []
    for start in range(0, len(docs), batch_size):
        print(f"[Search-R1] Dense embedding {min(start + batch_size, len(docs))}/{len(docs)}", flush=True)
        batch = docs[start:start + batch_size]
        texts = [f"passage: {x.get('text', x.get('contents', ''))}" for x in batch]
        tokens = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model(**tokens)
            emb = torch.nn.functional.normalize(mean_pool(out.last_hidden_state, tokens["attention_mask"]), dim=-1)
        vectors.append(emb.cpu().numpy().astype("float32"))
    matrix = np.concatenate(vectors, axis=0)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    os.makedirs(os.path.dirname(output_index) or ".", exist_ok=True)
    faiss.write_index(index, output_index)
    with open(manifest_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs):
            row = {"doc_id": str(doc.get("doc_id", i)), "title": doc.get("title", ""), "text": doc.get("text", doc.get("contents", ""))}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"documents": len(docs), "dimension": int(matrix.shape[1])}

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True)
    p.add_argument("--output-index", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", default="intfloat/e5-base-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=256)
    a = p.parse_args()
    print(build_dense(a.corpus, a.output_index, a.manifest, a.model, a.batch_size, a.max_length))
