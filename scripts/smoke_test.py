#!/usr/bin/env python3
import argparse
import json
import os
from search_r1_refine.rl.reward import score_trajectory
from search_r1_refine.retrieval.rrf import reciprocal_rank_fusion

p = argparse.ArgumentParser()
p.add_argument("--data-dir", default=None)
args = p.parse_args()
fused = reciprocal_rank_fusion([[{"doc_id": "a"}, {"doc_id": "b"}], [{"doc_id": "b"}, {"doc_id": "c"}]], topk=3)
assert fused[0]["doc_id"] == "b"
score = score_trajectory("<think>need evidence</think><search>capital</search><information>Paris is capital</information><answer>Paris</answer>", ["Paris"])
assert score["answer"] > 0 and score["total"] > 0
if args.data_dir:
    assert os.path.isdir(args.data_dir)
print(json.dumps({"status": "ok", "rrf": fused, "reward": score}, ensure_ascii=False))
