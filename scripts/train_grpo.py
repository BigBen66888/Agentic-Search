#!/usr/bin/env python3
import argparse
import importlib.util
import os
import shlex
import subprocess
import sys
import time

def log(message):
    print(f"[Search-R1][{time.strftime('%H:%M:%S')}] {message}", flush=True)

def check_environment(data_dir, n_gpus):
    log("阶段 1/4：检查 veRL 多卡环境")
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("未发现 torch，请先阅读 docs/从零安装与运行.md 安装 GPU 版 torch")
    import torch
    visible = torch.cuda.device_count()
    log(f"torch={torch.__version__}; CUDA={torch.cuda.is_available()}; visible_gpus={visible}")
    if n_gpus > 0 and visible < n_gpus:
        raise RuntimeError(f"需要 {n_gpus} 张 GPU，但只发现 {visible} 张")
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("未发现 veRL，请安装 requirements/03-verl-training.txt 或参考项目环境")
    train = os.path.join(data_dir, "processed", "train.parquet")
    val = os.path.join(data_dir, "processed", "eval.parquet")
    for path in (train, val):
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少训练文件：{path}，请先运行 scripts/prepare_data.py")
    log(f"数据检查通过：{train}；{val}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--reward-mode", choices=["binary", "multi"], default="multi")
    p.add_argument("--n-gpus", type=int, default=1)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-dir", default="artifacts/training")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.execute and args.dry_run:
        raise SystemExit("--execute 和 --dry-run 不能同时使用")
    if args.execute:
        check_environment(args.data_dir, args.n_gpus)
    else:
        log("阶段 1/4：dry-run，不检查本机 torch/veRL，可在服务器上执行")
    train = f"{args.data_dir}/processed/train.parquet"
    val = f"{args.data_dir}/processed/eval.parquet"
    os.makedirs(args.log_dir, exist_ok=True)
    command = ["python", "-m", "verl.trainer.main_ppo_format",
        f"data.train_files={train}", f"data.val_files={val}",
        "algorithm.adv_estimator=grpo", f"actor_rollout_ref.model.path={args.model}",
        "actor_rollout_ref.rollout.n_agent=5", "max_turns=4",
        f"retriever.url={args.retriever_url}", f"+reward_model.mode={args.reward_mode}",
        f"trainer.n_gpus_per_node={args.n_gpus}", f"trainer.nnodes={args.nnodes}",
        "actor_rollout_ref.actor.use_kl_loss=true", "actor_rollout_ref.actor.state_masking=true",
        "trainer.logger=['wandb']", f"trainer.default_local_dir={args.log_dir}/{args.reward_mode}"]
    if args.max_steps is not None:
        command.append(f"trainer.total_training_steps={args.max_steps}")
    log("阶段 2/4：生成 veRL GRPO 多卡命令")
    log(" ".join(shlex.quote(x) for x in command))
    if not args.execute:
        log("阶段 3/4：dry-run 完成；确认命令后加 --execute")
        return
    log("阶段 3/4：启动 veRL/Ray 多卡训练，实时输出子进程日志")
    if args.execute:
        env = os.environ.copy()
        env.setdefault("TOKENIZERS_PARALLELISM", "true")
        env.setdefault("NCCL_DEBUG", "WARN")
        result = subprocess.run(command, check=False, env=env)
        log(f"阶段 4/4：veRL 进程退出码={result.returncode}")
        if result.returncode != 0:
            raise SystemExit(result.returncode)

if __name__ == "__main__":
    main()
