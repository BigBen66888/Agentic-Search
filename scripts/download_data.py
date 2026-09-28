#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Download QA datasets, the 21M Wikipedia corpus and model weights.

Everything lands in an external ``--data-dir``; nothing is written into the
repository. Each artefact is optional so a partial run is still useful.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

SOURCES = ["nq", "hotpotqa"]


def snapshot(repo_id, target, repo_type=None, token=None, files=None):
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=repo_id, repo_type=repo_type, local_dir=str(target),
                             token=token, allow_patterns=files)


def main():
    p = argparse.ArgumentParser(description="Download datasets / corpus / models")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset-name", default="RUC-NLPIR/FlashRAG_datasets")
    p.add_argument("--corpus-repo", default="PeterJinGo/wiki-18-corpus")
    p.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--compare-model", default=None,
                   help="参考对比模型，例如 Qwen/Qwen3-4B；名字写错会在下载时报错")
    p.add_argument("--sources", nargs="+", default=SOURCES)
    p.add_argument("--qa-only", action="store_true")
    p.add_argument("--corpus", action="store_true")
    p.add_argument("--models", action="store_true")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--hf-endpoint", default=None, help="例如 https://hf-mirror.com")
    args = p.parse_args()

    if args.hf_endpoint:
        import os

        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_name": args.dataset_name, "corpus_repo": args.corpus_repo, "sources": args.sources}
    started = time.time()

    if args.qa_only or not (args.corpus or args.models):
        print(f"[Search-R1] 下载 QA 数据集 -> {root / 'qa'}", flush=True)
        raw = root / "raw" / "FlashRAG_datasets"
        snapshot(args.dataset_name, raw, repo_type="dataset", token=args.hf_token)
        found = {}
        for index, source in enumerate(args.sources, 1):
            destination = root / "qa" / source
            destination.mkdir(parents=True, exist_ok=True)
            matches = list(raw.rglob(f"{source}/*.jsonl")) + list(raw.rglob(f"{source}/*.json"))
            for source_file in matches:
                shutil.copy2(source_file, destination / f"{source_file.stem}{source_file.suffix}")
            found[source] = sorted(x.name for x in destination.iterdir()) if destination.exists() else []
            print(f"[Search-R1] QA [{index}/{len(args.sources)}] {source}: {found[source]}", flush=True)
        manifest["qa_files"] = found

    if args.corpus:
        out = root / "corpus" / "wiki-18"
        print(f"[Search-R1] 下载 Wikipedia 语料 {args.corpus_repo} -> {out}", flush=True)
        snapshot(args.corpus_repo, out, repo_type="dataset", token=args.hf_token)
        jsonl_files = sorted(out.rglob("*.jsonl"))
        manifest["corpus_files"] = [str(x) for x in jsonl_files]
        print(f"[Search-R1] 语料文件：{[x.name for x in jsonl_files]}", flush=True)

    if args.models:
        targets = [("embedding", args.embedding_model, root / "models" / args.embedding_model.split("/")[-1]),
                   ("base", args.base_model, root / "models" / args.base_model.split("/")[-1])]
        if args.compare_model:
            targets.append(("compare", args.compare_model,
                            root / "models" / args.compare_model.split("/")[-1]))
        for index, (kind, repo_id, target) in enumerate(targets, 1):
            print(f"[Search-R1] 模型 [{index}/{len(targets)}] {kind} {repo_id} -> {target}", flush=True)
            snapshot(repo_id, target, token=args.hf_token)
        manifest["models"] = {kind: {"repo": repo_id, "path": str(path)}
                              for kind, repo_id, path in targets}

    with open(root / "download_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"[Search-R1] 下载完成，用时 {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
