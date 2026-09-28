#!/usr/bin/env python3
"""One-command pipeline, in the exact experiment order requested.

    download -> indices -> prepare(+augment) -> sft-data
    -> eval-baseline -> sft-train -> eval-sft
    -> grpo-binary -> eval-grpo-binary
    -> grpo-multi  -> eval-grpo-multi
    -> eval-compare -> eval-single-way -> report

On a single AutoDL box the retrieval service starts after index construction
(CPU by default). One service handles both retrieval and the dense-only
ablation through the ?ways=dense query parameter.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

HERE = os.path.dirname(os.path.abspath(__file__))

STAGES = [
    "download", "indices", "prepare", "sft-data",
    "eval-baseline", "sft-train", "eval-sft",
    "grpo-binary", "eval-grpo-binary",
    "grpo-multi", "eval-grpo-multi",
    "eval-compare", "eval-single-way",
]


def run(command, stage):
    print(f"\n[Search-R1] ===== 阶段 {stage} 开始 =====", flush=True)
    print(f"[Search-R1] {' '.join(command)}", flush=True)
    started = time.time()
    result = subprocess.run(command, check=False)
    elapsed = time.time() - started
    print(f"[Search-R1] ===== 阶段 {stage} 结束，退出码={result.returncode}，用时 {elapsed:.1f}s =====\n",
          flush=True)
    if result.returncode != 0:
        raise SystemExit(f"阶段 {stage} 失败，退出码 {result.returncode}")
    return result


def health_url(retrieve_url):
    parts = urlsplit(retrieve_url)
    if parts.scheme not in ("http", "https") or not parts.netloc or parts.path != "/retrieve":
        raise ValueError(f"检索地址须为 http(s)://主机:端口/retrieve，收到 {retrieve_url!r}")
    return urlunsplit((parts.scheme, parts.netloc, "/health", "", ""))


def dense_url(retrieve_url):
    parts = urlsplit(retrieve_url)
    query = dict(parse_qsl(parts.query))
    query["ways"] = "dense"
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), ""))


def wait_health(url, expected_ways=None, timeout=1800, interval=10, process=None):
    import requests

    deadline = time.time() + timeout
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            return None
        try:
            response = requests.get(url, timeout=5)
            if response.ok:
                info = response.json()
                if info.get("status") == "ok" and (
                    expected_ways is None or set(info.get("ways", [])) == set(expected_ways)
                ):
                    return info
        except Exception:
            pass
        time.sleep(interval)
    return None


def start_retrievers(args):
    """Start one service; request-level ways selects the dense ablation."""
    command = [sys.executable, os.path.join(HERE, "serve_retriever.py"),
               "--data-dir", args.data_dir, "--port", str(args.port), "--ways", "both"]
    if args.retriever_cpu:
        command.append("--cpu")
    else:
        command += ["--gpu-id", str(args.retriever_gpu)]
    log_dir = os.path.join(args.data_dir, "artifacts", "retriever_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"retriever_{args.port}.log")
    with open(log_path, "w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
    info = wait_health(f"http://127.0.0.1:{args.port}/health",
                       expected_ways=["dense", "bm25"], process=process)
    if not info:
        stop_retrievers([process])
        raise RuntimeError(f"检索服务未就绪，请查看 {log_path}")
    print(f"[Search-R1] 检索服务就绪：{json.dumps(info, ensure_ascii=False)}", flush=True)
    return [process]


def stop_retrievers(processes):
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--compare-model", default="Qwen/Qwen3.5-4B")
    p.add_argument("--teacher-model", default="deepseek-4.1-flash")
    p.add_argument("--augment-model", default="deepseek-4.1-flash")
    p.add_argument("--embedding-model", default="intfloat/e5-base-v2")
    p.add_argument("--sft-target", type=int, default=2000)
    p.add_argument("--sft-epochs", type=float, default=3.0)
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument("--grpo-steps", type=int, default=None)
    p.add_argument("--fifo-mix-ratio", type=float, default=0.10)
    p.add_argument("--max-samples", type=int, default=None, help="评测抽样上限，用于快速验证")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--augment", action="store_true", default=True)
    p.add_argument("--no-augment", dest="augment", action="store_false")
    p.add_argument("--hf-endpoint", default=None)
    p.add_argument("--retriever-url", default=None, help="外部检索机地址；留空则本机启动")
    p.add_argument("--no-start-retriever", action="store_true")
    p.add_argument("--retriever-cpu", action="store_true", default=True)
    p.add_argument("--retriever-gpu-mode", dest="retriever_cpu", action="store_false")
    p.add_argument("--retriever-gpu", type=int, default=0)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--from-stage", default=STAGES[0], choices=STAGES)
    p.add_argument("--to-stage", default=STAGES[-1], choices=STAGES)
    p.add_argument("--skip-download", action="store_true")
    args = p.parse_args()

    data_dir = args.data_dir
    py = sys.executable
    start, end = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if start > end:
        raise SystemExit("--from-stage 不能晚于 --to-stage")

    def active(stage):
        return start <= STAGES.index(stage) <= end

    external_retriever = bool(args.retriever_url) or args.no_start_retriever
    if args.no_start_retriever and not args.retriever_url:
        raise SystemExit("--no-start-retriever 需要 --retriever-url")
    both_url = args.retriever_url or f"http://127.0.0.1:{args.port}/retrieve"
    single_url = dense_url(both_url)

    retrievers = []
    needs_retriever = any(active(s) for s in
                          ("sft-data", "eval-baseline", "eval-sft", "grpo-binary",
                           "grpo-multi", "eval-compare", "eval-single-way"))
    try:
        if active("download") and not args.skip_download:
            command = [py, os.path.join(HERE, "download_data.py"), "--data-dir", data_dir,
                       "--qa-only", "--corpus", "--models", "--sources", "nq", "hotpotqa",
                       "--base-model", args.base_model, "--embedding-model", args.embedding_model]
            if args.compare_model:
                command += ["--compare-model", args.compare_model]
            if args.hf_endpoint:
                command += ["--hf-endpoint", args.hf_endpoint]
            run(command, "download")

        if active("indices"):
            run([py, os.path.join(HERE, "build_indices.py"), "--data-dir", data_dir,
                 "--dense", "--bm25", "--model", args.embedding_model], "indices")

        if needs_retriever:
            if external_retriever:
                info = wait_health(health_url(both_url), expected_ways=["dense", "bm25"],
                                   timeout=30, interval=2)
                if not info:
                    raise RuntimeError(f"检索服务未就绪或召回通道不符：{both_url}")
                if active("eval-single-way"):
                    import requests
                    response = requests.post(single_url, json={"queries": [], "topk": 1}, timeout=10)
                    response.raise_for_status()
                    if response.json().get("ways") != "dense":
                        raise RuntimeError(f"检索服务不支持 dense 单路切换：{single_url}")
            else:
                retrievers = start_retrievers(args)

        if active("prepare"):
            command = [py, os.path.join(HERE, "prepare_data.py"), "--data-dir", data_dir]
            if args.augment:
                command += ["--augment", "--augment-model", args.augment_model]
            run(command, "prepare")

        if active("sft-data"):
            run([py, os.path.join(HERE, "generate_sft_data.py"), "--data-dir", data_dir,
                 "--teacher-model", args.teacher_model, "--retriever-url", both_url,
                 "--target", str(args.sft_target)], "sft-data")
            run([py, os.path.join(HERE, "update_fifo.py"), "--data-dir", data_dir, "--from-sft"],
                "sft-data")

        eval_command = [py, os.path.join(HERE, "compare_models.py"), "--data-dir", data_dir,
                        "--retriever-url", both_url, "--strict-missing"]
        if args.max_samples:
            eval_command += ["--max-samples", str(args.max_samples)]
        if args.judge:
            eval_command.append("--judge")

        if active("eval-baseline"):
            run(eval_command + ["--models", f"base={args.base_model}", "--only", "base"],
                "eval-baseline")

        if active("sft-train"):
            run([py, "-m", "torch.distributed.run", "--nproc_per_node", str(args.n_gpus),
                 os.path.join(HERE, "train_sft.py"), "--data-dir", data_dir,
                 "--model", args.base_model, "--epochs", str(args.sft_epochs)], "sft-train")

        if active("eval-sft"):
            run(eval_command + ["--models", f"sft={os.path.join(data_dir, 'models', 'sft')}",
                                "--only", "sft"], "eval-sft")

        if active("grpo-binary"):
            command = [py, os.path.join(HERE, "train_grpo.py"), "--data-dir", data_dir,
                       "--n-gpus", str(args.n_gpus), "--reward-mode", "binary",
                       "--run-name", "grpo_binary", "--retriever-url", both_url,
                       "--fifo-mix-ratio", str(args.fifo_mix_ratio), "--execute"]
            if args.grpo_steps:
                command += ["--max-steps", str(args.grpo_steps)]
            run(command, "grpo-binary")

        if active("eval-grpo-binary"):
            run(eval_command + ["--models",
                                f"grpo_binary={os.path.join(data_dir, 'models', 'grpo_binary')}",
                                "--only", "grpo_binary"], "eval-grpo-binary")

        if active("grpo-multi"):
            command = [py, os.path.join(HERE, "train_grpo.py"), "--data-dir", data_dir,
                       "--n-gpus", str(args.n_gpus), "--reward-mode", "multi",
                       "--run-name", "grpo_multi", "--retriever-url", both_url,
                       "--fifo-mix-ratio", str(args.fifo_mix_ratio), "--execute"]
            if args.grpo_steps:
                command += ["--max-steps", str(args.grpo_steps)]
            run(command, "grpo-multi")

        if active("eval-grpo-multi"):
            run(eval_command + ["--models",
                                f"grpo_multi={os.path.join(data_dir, 'models', 'grpo_multi')}",
                                "--only", "grpo_multi"], "eval-grpo-multi")

        if active("eval-compare"):
            run(eval_command + ["--models", f"compare={args.compare_model}",
                                "--only", "compare"], "eval-compare")

        if active("eval-single-way"):
            run(eval_command + ["--models",
                                f"grpo_multi_single={os.path.join(data_dir, 'models', 'grpo_multi')}",
                                "--variant-retriever", f"grpo_multi_single={single_url}",
                                "--only", "grpo_multi_single"], "eval-single-way")

        print("[Search-R1] 流水线执行完成，报告见 "
              f"{os.path.join(data_dir, 'artifacts', 'report', 'comparison_report.md')}", flush=True)
    finally:
        stop_retrievers(retrievers)


if __name__ == "__main__":
    main()
