#!/usr/bin/env python3
"""GRPO training on 4x RTX 4090 through veRL + Ray.

Starts from the SFT checkpoint. Two things are wired explicitly:

* reward mode -> ``SEARCH_REWARD_MODE`` env var *and* the veRL config, so the
  0/1-reward run and the multi-level-reward run can no longer collapse into one
* FIFO replay -> ``--fifo-mix-ratio`` of the training prompts are replaced by
  prompts the policy already solved, keeping the solved behaviour from decaying
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time


def log(message):
    print(f"[Search-R1][{time.strftime('%H:%M:%S')}] {message}", flush=True)


def check_environment(data_dir, n_gpus, train_file):
    log("阶段 1/4：检查训练环境")
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError("未发现 torch，请先安装 requirements/01-torch-cu121.txt")
    import torch

    visible = torch.cuda.device_count()
    log(f"torch={torch.__version__}; cuda={torch.cuda.is_available()}; visible_gpus={visible}")
    if n_gpus > 0 and visible < n_gpus:
        raise RuntimeError(f"需要 {n_gpus} 张 GPU，实际可见 {visible} 张")
    if importlib.util.find_spec("verl") is None:
        raise RuntimeError("未发现 veRL，请安装 requirements/03-verl-training.txt")
    for path in (train_file, os.path.join(data_dir, "processed", "eval.parquet")):
        if not os.path.exists(path):
            raise FileNotFoundError(f"缺少 {path}，请先运行 scripts/prepare_data.py")
    log("环境检查通过")


def build_replay_pool(data_dir, fifo_path, ratio, seed):
    """Replace ``ratio`` of the training prompts with solved prompts from FIFO."""
    from search_r1_refine.data.schema import read_jsonl, write_jsonl
    from search_r1_refine.rl.fifo import SuccessFIFO, build_mixed_pool

    pool_path = os.path.join(data_dir, "processed", "train_pool.jsonl")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"缺少 {pool_path}，请先运行 scripts/prepare_data.py")
    pool = read_jsonl(pool_path)
    fifo = SuccessFIFO.load(fifo_path)
    mixed = build_mixed_pool(pool, fifo, ratio=ratio, seed=seed)
    out_jsonl = os.path.join(data_dir, "processed", "train_pool_mixed.jsonl")
    write_jsonl(mixed["rows"], out_jsonl)
    out_parquet = os.path.join(data_dir, "processed", "train_pool_mixed.parquet")
    try:
        import pandas as pd

        pd.DataFrame(mixed["rows"]).to_parquet(out_parquet, index=False)
    except Exception as exc:
        log(f"混合数据 Parquet 导出失败：{exc}，回退到 train_pool.parquet")
        return os.path.join(data_dir, "processed", "train_pool.parquet"), mixed
    stats = {"fifo_size": len(fifo), "replay_rows": mixed["replay_rows"],
             "pool_rows": mixed["pool_rows"], "ratio": ratio, "output": out_parquet}
    log(f"FIFO 混入：{json.dumps(stats, ensure_ascii=False)}")
    return out_parquet, stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", default=None, help="默认使用 SFT checkpoint")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B")
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--reward-mode", choices=["binary", "multi"], default="multi")
    p.add_argument("--run-name", default=None, help="默认用 reward-mode 命名")
    p.add_argument("--n-gpus", type=int, default=4)
    p.add_argument("--nnodes", type=int, default=1)
    p.add_argument("--rollout-n", type=int, default=5)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-dir", default="artifacts/training")
    p.add_argument("--fifo-path", default=None, help="默认 <data-dir>/processed/fifo.jsonl")
    p.add_argument("--fifo-mix-ratio", type=float, default=0.10)
    p.add_argument("--no-fifo", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--upstream-verl", action="store_true",
                   help="使用 actor_rollout_ref.rollout.n 而非 Search-R1 fork 的 n_agent")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()

    if args.execute and args.dry_run:
        raise SystemExit("--execute 和 --dry-run 不能同时使用")

    run_name = args.run_name or args.reward_mode
    model = args.model or os.path.join(args.data_dir, "models", "sft")
    val = os.path.join(args.data_dir, "processed", "eval.parquet")
    log_dir = os.path.join(args.data_dir, args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    fifo_path = args.fifo_path or os.path.join(args.data_dir, "processed", "fifo.jsonl")
    train = os.path.join(args.data_dir, "processed", "train_pool.parquet")
    fifo_stats = {"enabled": False}
    if not args.no_fifo and args.fifo_mix_ratio > 0:
        if os.path.exists(fifo_path):
            train, fifo_stats = build_replay_pool(args.data_dir, fifo_path, args.fifo_mix_ratio, args.seed)
            fifo_stats["enabled"] = True
        else:
            log(f"未找到 FIFO 缓冲 {fifo_path}，本次不做成功轨迹混入（可用 scripts/update_fifo.py 生成）")

    if args.execute:
        check_environment(args.data_dir, args.n_gpus, train)
        if not os.path.exists(model):
            raise FileNotFoundError(f"模型路径不存在：{model}，请先运行 SFT 或指定 --model")
    else:
        log("阶段 1/4：dry-run，不检查本机 torch/veRL")

    custom_reward = os.path.abspath(os.path.join(os.path.dirname(__file__), "verl_custom_reward.py"))
    rollout_key = "actor_rollout_ref.rollout.n" if args.upstream_verl else "actor_rollout_ref.rollout.n_agent"
    command = [sys.executable, "-m", "verl.trainer.main_ppo",
               f"data.train_files={train}", f"data.val_files={val}",
               "algorithm.adv_estimator=grpo", f"actor_rollout_ref.model.path={model}",
               f"{rollout_key}={args.rollout_n}", f"max_turns={args.max_turns}",
               f"retriever.url={args.retriever_url}", f"+reward_model.mode={args.reward_mode}",
               f"trainer.n_gpus_per_node={args.n_gpus}", f"trainer.nnodes={args.nnodes}",
               "actor_rollout_ref.actor.use_kl_loss=true",
               "actor_rollout_ref.actor.state_masking=true",
               "actor_rollout_ref.model.enable_gradient_checkpointing=true",
               "actor_rollout_ref.actor.fsdp_config.param_offload=true",
               "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
               f"custom_reward_function.path={custom_reward}",
               "custom_reward_function.name=compute_score",
               f"trainer.default_local_dir={log_dir}"]
    if args.max_steps is not None:
        command.append(f"trainer.total_training_steps={args.max_steps}")

    log("阶段 2/4：生成 veRL GRPO 命令")
    log(" ".join(shlex.quote(x) for x in command))
    with open(os.path.join(log_dir, "train_command.txt"), "w", encoding="utf-8") as f:
        f.write(" ".join(shlex.quote(x) for x in command) + "\n")
    with open(os.path.join(log_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"reward_mode": args.reward_mode, "run_name": run_name, "model": model,
                   "train_files": train, "fifo": fifo_stats, "seed": args.seed,
                   "rollout_n": args.rollout_n, "max_turns": args.max_turns},
                  f, ensure_ascii=False, indent=2)
    if not args.execute:
        log("阶段 3/4：dry-run 完成；确认命令后加 --execute")
        return

    log("阶段 3/4：启动 veRL/Ray 多卡训练")
    env = os.environ.copy()
    env["SEARCH_REWARD_MODE"] = args.reward_mode
    env["SEARCH_REWARD_MAX_TURNS"] = str(args.max_turns)
    env["SEARCH_GRPO_SUCCESS_WEIGHT"] = "1.5"
    env["SEARCH_GRPO_FAILURE_WEIGHT"] = "0.6"
    env["SEARCH_GRPO_CLIP_SIGMA"] = "5.0"
    env.setdefault("TOKENIZERS_PARALLELISM", "true")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("VERL_ATTN_IMPL", "sdpa")
    result = subprocess.run(command, check=False, env=env)
    log(f"阶段 4/4：veRL 进程退出码={result.returncode}")
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    expected = os.path.join(args.data_dir, "models", run_name)
    log(f"把最终 checkpoint 链接到 {expected} 后即可参与评测：")
    log(f"    ln -sfn \"$(ls -td {log_dir}/global_step_* | head -1)/actor\" {expected}")


if __name__ == "__main__":
    main()
