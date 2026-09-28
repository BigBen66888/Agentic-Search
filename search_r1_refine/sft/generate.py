"""Sample high-scoring teacher trajectories and build the SFT cold-start set.

Pipeline: pick prompts from the train pool -> drive the teacher through the real
RRF retriever -> score every trajectory with the same multi-level reward used in
GRPO -> keep the top ``--target`` (default 2000) trajectories.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from search_r1_refine.data.schema import read_jsonl, write_jsonl
from search_r1_refine.rl.reward import score_trajectory

from .teacher import ChatClient, RetrieverClient, run_teacher_episode, trajectory_to_training_row


def balance_by_source(rows: List[Dict], limit: int, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    by_source: Dict[str, List[Dict]] = {}
    for row in rows:
        by_source.setdefault(row.get("source", "unknown"), []).append(row)
    for items in by_source.values():
        rng.shuffle(items)
    picked, sources = [], sorted(by_source)
    if not sources:
        return []
    per_source = max(limit // len(sources), 1)
    for source in sources:
        picked.extend(by_source[source][:per_source])
    rng.shuffle(picked)
    return picked[:limit]


def generate(args) -> Dict:
    pool = read_jsonl(args.train_pool)
    if not pool:
        raise FileNotFoundError(f"训练池为空：{args.train_pool}，请先运行 scripts/prepare_data.py")
    prompts = balance_by_source(pool, args.max_attempts, args.seed)
    print(f"[Search-R1] SFT 采样：候选 {len(prompts)} 条，目标保留 {args.target} 条", flush=True)

    client = ChatClient(model=args.teacher_model, base_url=args.api_base, api_key=args.api_key,
                        temperature=args.temperature, max_tokens=args.max_tokens)
    retriever = RetrieverClient(url=args.retriever_url, topk=args.topk,
                                max_chars=args.max_chars, timeout=args.timeout)

    collected: List[Dict] = []
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(run_teacher_episode, client, retriever,
                            row.get("prompt_text") or row.get("question", ""),
                            args.max_turns, args.temperature): row
            for row in prompts
        }
        for done, future in enumerate(as_completed(futures), 1):
            row = futures[future]
            try:
                trajectory = future.result()
            except Exception as exc:
                failures += 1
                if failures <= 5:
                    print(f"[Search-R1] 采样失败：{exc}", flush=True)
                continue
            parts = score_trajectory(trajectory["text"], row.get("answers", []),
                                     row.get("supporting_facts"), mode="multi",
                                     max_turns=args.max_turns)
            collected.append(trajectory_to_training_row(row, trajectory, parts))
            if done % 50 == 0:
                print(f"[Search-R1] 采样进度 {done}/{len(prompts)}，已收集 {len(collected)}", flush=True)

    kept = [x for x in collected
            if x["reward"].get("answer", 0) >= args.min_answer_f1
            and x["reward_total"] >= args.min_reward
            and x["search_count"] <= args.max_turns]
    kept.sort(key=lambda x: (-x["reward_total"], x["search_count"]))
    kept = kept[:args.target]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    write_jsonl(kept, args.output)
    stats = {
        "attempted": len(prompts),
        "collected": len(collected),
        "failures": failures,
        "kept": len(kept),
        "target": args.target,
        "min_reward": args.min_reward,
        "min_answer_f1": args.min_answer_f1,
        "teacher_model": args.teacher_model,
        "avg_reward": round(sum(x["reward_total"] for x in kept) / max(len(kept), 1), 4),
        "avg_searches": round(sum(x["search_count"] for x in kept) / max(len(kept), 1), 3),
        "by_source": {s: sum(1 for x in kept if x.get("source") == s)
                      for s in sorted({x.get("source") for x in kept})},
        "output": args.output,
    }
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)
    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate SFT cold-start trajectories")
    p.add_argument("--train-pool", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--stats", default=None)
    p.add_argument("--teacher-model",
                   default=os.environ.get("TEACHER_MODEL", "deepseek-4.1-flash"))
    p.add_argument("--api-base", default=os.environ.get("LLM_API_BASE"))
    p.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"))
    p.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    p.add_argument("--target", type=int, default=2000, help="保留的高分轨迹条数")
    p.add_argument("--max-attempts", type=int, default=6000)
    p.add_argument("--max-turns", type=int, default=4)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--max-chars", type=int, default=1200)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--min-reward", type=float, default=0.8)
    p.add_argument("--min-answer-f1", type=float, default=0.9)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--seed", type=int, default=0)
    return p


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if not args.stats:
        args.stats = os.path.join(os.path.dirname(args.output), "sft_stats.json")
    generate(args)
