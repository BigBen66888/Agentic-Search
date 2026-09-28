#!/usr/bin/env python3
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
"""Dependency-light smoke test.

Runs on a laptop with no GPU, no torch, no faiss: it exercises the pure-python
core (fusion, reward, metrics, dataset selection, report) end to end and prints
a single JSON status. Heavy stages (index build, SFT, GRPO) are validated by
scripts/check_env.py on the GPU box instead.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile

from search_r1_refine.data.prepare import prepare, split_is_usable
from search_r1_refine.evaluation.rank import mrr_at_k, ndcg_at_k, recall_at_k
from search_r1_refine.evaluation.report import build_comparison, to_markdown
from search_r1_refine.retrieval.rrf import reciprocal_rank_fusion
from search_r1_refine.rl.fifo import SuccessFIFO, build_mixed_pool
from search_r1_refine.rl.reward import score_trajectory

CHECKS = []


def check(name):
    def decorator(func):
        CHECKS.append((name, func))
        return func

    return decorator


@check("rrf_fusion")
def _rrf():
    fused = reciprocal_rank_fusion([[{"doc_id": "a"}, {"doc_id": "b"}],
                                    [{"doc_id": "b"}, {"doc_id": "c"}]], topk=3)
    assert [x["doc_id"] for x in fused] == ["b", "a", "c"], fused
    return {"order": [x["doc_id"] for x in fused]}


@check("reward")
def _reward():
    text = ("<think>need evidence</think><search>capital of france</search>"
            "<information>Paris is the capital of France.</information>"
            "<think>found it</think><answer>Paris</answer>")
    multi = score_trajectory(text, ["Paris"])
    binary = score_trajectory(text, ["Paris"], mode="binary")
    wrong = score_trajectory(text, ["London"], mode="binary")
    assert multi["answer"] > 0 and multi["total"] > 0
    assert binary["total"] == 1.0 and wrong["total"] == 0.0
    return {"multi": multi, "binary": binary["total"], "wrong": wrong["total"]}


@check("rank_metrics")
def _rank():
    gold = {"paris", "france"}
    ranked = ["london", "paris", "berlin", "france"]
    assert recall_at_k(ranked, gold, 10) == 1.0
    assert mrr_at_k(ranked, gold, 10) == 0.5
    assert ndcg_at_k(ranked, gold, 10) > 0 and ndcg_at_k(ranked, gold, 10) <= 1.0
    assert mrr_at_k(["london"], gold, 10) == 0.0
    return {"recall@10": recall_at_k(ranked, gold, 10),
            "mrr@10": mrr_at_k(ranked, gold, 10),
            "ndcg@10": ndcg_at_k(ranked, gold, 10)}


@check("strict_official_test_selection")
def _dataset_selection():
    tmp = tempfile.mkdtemp(prefix="sr1_smoke_")
    try:
        cases = {
            "nq": {"test": [{"question": "Who wrote Hamlet?", "golden_answers": ["Shakespeare"]}] * 12},
            "triviaqa": {"test": [{"question": "Capital of France?", "golden_answers": ["Paris"]}] * 12},
            "hotpotqa": {"dev": [{"question": "Who is X?", "golden_answers": []}] * 12},
            "popqa": {"dev": [{"question": "Who is Y?", "golden_answers": ["x"]}] * 12},
            "bamboogle": {"test": [{"question": "Who is Z?", "golden_answers": ["y"]}] * 3},
        }
        for source, splits in cases.items():
            directory = os.path.join(tmp, "qa", source)
            os.makedirs(directory, exist_ok=True)
            for split, rows in splits.items():
                with open(os.path.join(directory, f"{split}.jsonl"), "w", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")

        class Args:
            dataset_name = "mock"
            sources = ["nq", "triviaqa", "hotpotqa", "popqa", "bamboogle"]
            data_dir = tmp
            output_dir = os.path.join(tmp, "processed")
            eval_split = "test"
            eval_splits = {"hotpotqa": "dev"}
            min_answer_rate = 0.9
            min_rows = 10
            min_chars = 5
            max_chars = 2000
            require_intent = True
            allow_dev_fallback = False
            augment = False
            easy_max = 1
            medium_max = 3

        stats = prepare(Args())
        assert stats["eval_sources"] == ["nq", "triviaqa"], stats["eval_sources"]
        assert stats["dropped_sources"]["hotpotqa"].startswith("answers_missing"), stats["dropped_sources"]
        assert stats["dropped_sources"]["popqa"].startswith("split_missing"), stats["dropped_sources"]
        assert stats["dropped_sources"]["bamboogle"].startswith("too_few_rows"), stats["dropped_sources"]
        assert os.path.exists(os.path.join(Args.output_dir, "eval.jsonl"))
        return {"kept": stats["eval_sources"], "dropped": stats["dropped_sources"]}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("fifo_replay")
def _fifo():
    fifo = SuccessFIFO(max_size=100)
    for i in range(20):
        fifo.add({"sample_id": f"s{i}", "question": f"q{i}?", "answers": ["a"],
                  "success": True, "reward_total": 0.9})
    pool = [{"sample_id": f"p{i}", "question": "q?", "answers": ["a"]} for i in range(100)]
    mixed = build_mixed_pool(pool, fifo, ratio=0.10, seed=0)
    assert mixed["replay_rows"] == 10, mixed
    assert len(mixed["rows"]) == 100
    return {"fifo_size": len(fifo), "replay_rows": mixed["replay_rows"]}


@check("comparison_report")
def _report():
    summaries = {
        "base": {"overall": {"em": 0.20, "answer_f1": 0.30, "mrr@10": 0.25, "ndcg@10": 0.28,
                              "count": 10}, "by_source": {}},
        "sft": {"overall": {"em": 0.30, "answer_f1": 0.42, "mrr@10": 0.35, "ndcg@10": 0.40,
                             "count": 10}, "by_source": {}},
    }
    comparison = build_comparison(summaries, baseline="base")
    markdown = to_markdown(comparison)
    assert "sft" in markdown and "ΔEM" in markdown
    return {"variants": comparison["variants"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None, help="预留：GPU 机器上检查真实索引")
    args = parser.parse_args()
    results, failures = {}, []
    for name, func in CHECKS:
        try:
            results[name] = func()
            print(f"[Search-R1] smoke ok: {name}", flush=True)
        except Exception as exc:
            failures.append({"check": name, "error": repr(exc)})
            print(f"[Search-R1] smoke FAILED: {name}: {exc}", flush=True)
    status = {"status": "ok" if not failures else "failed", "checks": results, "failures": failures}
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
