#!/usr/bin/env python3
import argparse
import json
import shutil
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
    p.add_argument("--models", action="store_true", help="同时下载 embedding、训练和可选改写模型")
    p.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    p.add_argument("--train-model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--rewrite-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = p.parse_args()
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset_name": args.dataset_name, "sources": args.sources, "corpus_repo": args.corpus_repo}
    started = time.time()
    print(f"[Search-R1] 数据下载开始：目标目录={root}", flush=True)
    if not args.corpus or args.qa_only:
        from huggingface_hub import snapshot_download
        raw = root / "raw" / "FlashRAG_datasets"
        print(f"[Search-R1] 下载 QA 数据仓库到 {raw}", flush=True)
        snapshot_download(repo_id=args.dataset_name, repo_type="dataset", local_dir=str(raw))
        for i, source in enumerate(args.sources, 1):
            destination = root / "qa" / source
            destination.mkdir(parents=True, exist_ok=True)
            matches = list(raw.rglob(f"{source}/*.jsonl")) + list(raw.rglob(f"{source}/*.json"))
            for source_file in matches:
                split = source_file.stem
                target = destination / f"{split}{source_file.suffix}"
                shutil.copy2(source_file, target)
            print(f"[Search-R1] QA 数据 [{i}/{len(args.sources)}] {source}：{len(matches)} 个文件", flush=True)
    if args.corpus and not args.qa_only:
        from huggingface_hub import snapshot_download
        out = root / "corpus" / "wiki-18"
        print(f"[Search-R1] Wikipedia 语料开始下载：{args.corpus_repo}", flush=True)
        snapshot_download(repo_id=args.corpus_repo, repo_type="dataset", local_dir=str(out))
        manifest["corpus_dir"] = str(out)
    if args.models:
        from huggingface_hub import snapshot_download
        models = [(args.embedding_model, root / "models" / "e5-base-v2"),
                  (args.train_model, root / "models" / args.train_model.split("/")[-1]),
                  (args.rewrite_model, root / "models" / args.rewrite_model.split("/")[-1])]
        for i, (repo_id, target) in enumerate(models, 1):
            print(f"[Search-R1] 模型 [{i}/{len(models)}] {repo_id} -> {target}", flush=True)
            snapshot_download(repo_id=repo_id, local_dir=str(target))
        manifest["models"] = [str(path) for _, path in models]
    with open(root / "download_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[Search-R1] 数据下载阶段完成，用时 {time.time() - started:.1f}s", flush=True)

if __name__ == "__main__":
    main()
