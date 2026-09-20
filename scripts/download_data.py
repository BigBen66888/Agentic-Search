#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

SOURCES = ["nq","triviaqa","popqa","hotpotqa","2wikimultihopqa","musique","bamboogle"]

def main():
    p = argparse.ArgumentParser(description="Download datasets/corpus to an external directory.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--dataset-name", default="RUC-NLPIR/FlashRAG_datasets")
    p.add_argument("--corpus", action="store_true")
    p.add_argument("--qa-only", action="store_true")
    p.add_argument("--sources", nargs="+", default=SOURCES)
    p.add_argument("--corpus-repo", default="PeterJinGo/wiki-18-corpus")
    args = p.parse_args()
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_name": args.dataset_name, "sources": args.sources, "corpus_repo": args.corpus_repo}
    started = time.time()
    print(f"[Search-R1] 数据下载开始：目标目录={root}", flush=True)
    if not args.corpus or args.qa_only:
        from datasets import load_dataset
        for i, source in enumerate(args.sources, 1):
            print(f"[Search-R1] QA 数据 [{i}/{len(args.sources)}] {source}：开始", flush=True)
            load_dataset(args.dataset_name, source, trust_remote_code=True)
            print(f"[Search-R1] QA 数据 [{i}/{len(args.sources)}] {source}：完成", flush=True)
    if args.corpus and not args.qa_only:
        from huggingface_hub import snapshot_download
        out = root / "corpus" / "wiki-18"
        print(f"[Search-R1] Wikipedia 语料开始下载：{args.corpus_repo}", flush=True)
        snapshot_download(repo_id=args.corpus_repo, repo_type="dataset", local_dir=str(out))
        manifest["corpus_dir"] = str(out)
    with open(root / "download_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[Search-R1] 数据下载阶段完成，用时 {time.time() - started:.1f}s", flush=True)

if __name__ == "__main__":
    main()
