#!/usr/bin/env python3
"""Run model variants through the same evaluation and build the comparison report.

Each variant: start a vLLM agent service -> wait for /health -> evaluate on the
official test sets -> stop the service. Every variant may point at a different
retrieval service, which is how the single-way vs two-way ablation is measured.

Summaries already on disk are merged back into the report, so the script can be
called repeatedly as models finish training.

    python scripts/compare_models.py --data-dir /data/search_r1 \
        --models base=Qwen/Qwen2.5-3B sft=/data/search_r1/models/sft \
                 grpo_binary=/data/search_r1/models/grpo_binary \
                 grpo_multi=/data/search_r1/models/grpo_multi \
                 compare=Qwen/Qwen3.5-4B \
                 grpo_multi_single=/data/search_r1/models/grpo_multi \
        --variant-retriever grpo_multi_single=http://127.0.0.1:8001/retrieve
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))


def wait_healthy(url, timeout=1800, interval=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(url, timeout=5)
            if response.ok and response.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def start_agent(model, port, retriever_url, gpu_memory_utilization, max_turns, max_model_len,
                data_dir):
    command = [sys.executable, os.path.join(HERE, "serve_agent.py"),
               "--model", model, "--retriever-url", retriever_url,
               "--port", str(port), "--max-turns", str(max_turns),
               "--gpu-memory-utilization", str(gpu_memory_utilization),
               "--max-model-len", str(max_model_len)]
    log_dir = os.path.join(data_dir, "artifacts", "agent_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"agent_{port}.log")
    handle = open(log_path, "w", encoding="utf-8")
    return subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT), log_path


def evaluate_variant(data_dir, variant, retriever_url, port, max_samples, rank_k,
                     judge, judge_model, timeout):
    command = [sys.executable, os.path.join(HERE, "evaluate_online.py"),
               "--data-dir", data_dir, "--agent-url", f"http://127.0.0.1:{port}/generate",
               "--retriever-url", retriever_url, "--variant", variant,
               "--rank-k", str(rank_k), "--timeout", str(timeout)]
    if max_samples:
        command += ["--max-samples", str(max_samples)]
    if judge:
        command.append("--judge")
        if judge_model:
            command += ["--judge-model", judge_model]
    subprocess.run(command, check=True)
    path = os.path.join(data_dir, "artifacts", "eval", variant, "eval_summary.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_pairs(items, separator="="):
    out = {}
    for item in items or []:
        name, _, value = item.partition(separator)
        if not value:
            raise SystemExit(f"需要 name{separator}value 形式，收到 {item!r}")
        out[name.strip()] = value.strip()
    return out


def default_models(data_dir):
    models_dir = os.path.join(data_dir, "models")
    return {
        "base": "Qwen/Qwen2.5-3B",
        "sft": os.path.join(models_dir, "sft"),
        "grpo_binary": os.path.join(models_dir, "grpo_binary"),
        "grpo_multi": os.path.join(models_dir, "grpo_multi"),
        "compare": "Qwen/Qwen3.5-4B",
    }


def collect_existing(data_dir):
    summaries = {}
    for path in glob.glob(os.path.join(data_dir, "artifacts", "eval", "*", "eval_summary.json")):
        variant = os.path.basename(os.path.dirname(path))
        try:
            with open(path, encoding="utf-8") as f:
                summaries[variant] = json.load(f)
        except Exception:
            continue
    return summaries


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--models", nargs="+", default=None, help="name=model_path 列表")
    p.add_argument("--variant-retriever", nargs="*", default=None,
                   help="为特定变体指定检索服务，例如 grpo_multi_single=http://127.0.0.1:8001/retrieve")
    p.add_argument("--only", nargs="*", default=None, help="只跑这些变体")
    p.add_argument("--base-port", type=int, default=9100)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--rank-k", type=int, default=10)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--judge", action="store_true")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--baseline", default="base")
    p.add_argument("--no-merge", action="store_true", help="不合并历史评测结果")
    p.add_argument("--skip-missing", action="store_true", default=True,
                   help="模型不存在或加载失败时跳过该变体（默认开启）")
    p.add_argument("--strict-missing", dest="skip_missing", action="store_false")
    args = p.parse_args()

    models = parse_pairs(args.models) or default_models(args.data_dir)
    variant_retriever = parse_pairs(args.variant_retriever)
    if args.only:
        models = {k: v for k, v in models.items() if k in set(args.only)}

    summaries = {} if args.no_merge else collect_existing(args.data_dir)
    if summaries:
        print(f"[Search-R1] 合并已有评测结果：{sorted(summaries)}", flush=True)

    for index, (variant, model) in enumerate(models.items()):
        port = args.base_port + index
        retriever_url = variant_retriever.get(variant, args.retriever_url)
        print(f"[Search-R1] 变体 {variant} -> {model} (port {port}, retriever {retriever_url})", flush=True)
        process, log_path = start_agent(model, port, retriever_url,
                                        args.gpu_memory_utilization, args.max_turns,
                                        args.max_model_len, args.data_dir)
        try:
            if not wait_healthy(f"http://127.0.0.1:{port}/health", timeout=1800):
                message = f"agent 服务未就绪（{variant}），日志：{log_path}"
                if args.skip_missing:
                    print(f"[Search-R1] 跳过：{message}", flush=True)
                    continue
                raise RuntimeError(message)
            summaries[variant] = evaluate_variant(args.data_dir, variant, retriever_url, port,
                                                  args.max_samples, args.rank_k, args.judge,
                                                  args.judge_model, args.timeout)
        except subprocess.CalledProcessError as exc:
            if args.skip_missing:
                print(f"[Search-R1] 变体 {variant} 评测失败，跳过：{exc}", flush=True)
                continue
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
            time.sleep(15)

    if not summaries:
        raise SystemExit("没有任何变体评测成功")

    from search_r1_refine.evaluation.report import build_comparison, write_report

    comparison = build_comparison(summaries, baseline=args.baseline)
    out_dir = os.path.join(args.data_dir, "artifacts", "report")
    paths = write_report(comparison, out_dir)
    print(json.dumps({"variants": list(summaries), "report": paths}, ensure_ascii=False, indent=2), flush=True)
    print(f"[Search-R1] 对比报告：{paths['markdown']}", flush=True)


if __name__ == "__main__":
    main()
